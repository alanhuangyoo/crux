/**
 * Context compaction for long sessions.
 *
 * Pure functions for compaction logic. The session manager handles I/O,
 * and after compaction the session is reloaded.
 */

import type { AgentMessage, StreamFn, ThinkingLevel } from "@earendil-works/pi-agent-core";
import { contentText, type RetryCallbacks, type RetryPolicy, retryAssistantCall, uuidv7 } from "@earendil-works/pi-ai";
import type { AssistantMessage, Context, Model, SimpleStreamOptions, Usage } from "@earendil-works/pi-ai/compat";
import { completeSimple } from "@earendil-works/pi-ai/compat";
import { convertToLlm } from "../messages.ts";
import {
	buildSessionContext,
	type CompactionEntry,
	type SessionEntry,
	sessionEntryToContextMessages,
} from "../session-manager.ts";
import {
	computeFileLists,
	createFileOps,
	extractFileOpsFromMessage,
	type FileOperations,
	formatFileOperations,
	SUMMARIZATION_SYSTEM_PROMPT,
	serializeConversation,
} from "./utils.ts";

// ============================================================================
// File Operation Tracking
// ============================================================================

/** Details stored in CompactionEntry.details for file tracking */
export interface CompactionDetails {
	readFiles: string[];
	modifiedFiles: string[];
}

/**
 * Extract file operations from messages and previous compaction entries.
 */
function extractFileOperations(
	messages: AgentMessage[],
	entries: SessionEntry[],
	prevCompactionIndex: number,
): FileOperations {
	const fileOps = createFileOps();

	// Collect from previous compaction's details (if pi-generated)
	if (prevCompactionIndex >= 0) {
		const prevCompaction = entries[prevCompactionIndex] as CompactionEntry;
		if (!prevCompaction.fromHook && prevCompaction.details) {
			// fromHook field kept for session file compatibility
			const details = prevCompaction.details as CompactionDetails;
			if (Array.isArray(details.readFiles)) {
				for (const f of details.readFiles) fileOps.read.add(f);
			}
			if (Array.isArray(details.modifiedFiles)) {
				for (const f of details.modifiedFiles) fileOps.edited.add(f);
			}
		}
	}

	// Extract from tool calls in messages
	for (const msg of messages) {
		extractFileOpsFromMessage(msg, fileOps);
	}

	return fileOps;
}

// ============================================================================
// Message Extraction
// ============================================================================

/**
 * Extract AgentMessage from an entry if it produces one.
 * Returns undefined for entries that don't contribute to LLM context.
 */
function getMessageFromEntryForCompaction(entry: SessionEntry): AgentMessage | undefined {
	if (entry.type === "compaction") {
		return undefined;
	}
	return sessionEntryToContextMessages(entry)[0];
}

/** Result from compact() - SessionManager adds uuid/parentUuid when saving */
export interface CompactionResult<T = unknown> {
	summary: string;
	firstKeptEntryId: string;
	tokensBefore: number;
	estimatedTokensAfter?: number;
	/** Usage from the LLM call(s) that generated this summary, if available */
	usage?: Usage;
	/** Extension-specific data (e.g., ArtifactIndex, version markers for structured compaction) */
	details?: T;
}

function combineUsage(first: Usage, second: Usage): Usage {
	return {
		input: first.input + second.input,
		output: first.output + second.output,
		cacheRead: first.cacheRead + second.cacheRead,
		cacheWrite: first.cacheWrite + second.cacheWrite,
		...(first.cacheWrite1h !== undefined || second.cacheWrite1h !== undefined
			? { cacheWrite1h: (first.cacheWrite1h ?? 0) + (second.cacheWrite1h ?? 0) }
			: {}),
		...(first.reasoning !== undefined || second.reasoning !== undefined
			? { reasoning: (first.reasoning ?? 0) + (second.reasoning ?? 0) }
			: {}),
		totalTokens: first.totalTokens + second.totalTokens,
		cost: {
			input: first.cost.input + second.cost.input,
			output: first.cost.output + second.cost.output,
			cacheRead: first.cost.cacheRead + second.cost.cacheRead,
			cacheWrite: first.cost.cacheWrite + second.cost.cacheWrite,
			total: first.cost.total + second.cost.total,
		},
	};
}

// ============================================================================
// Types
// ============================================================================

export interface CompactionSettings {
	enabled: boolean;
	reserveTokens: number;
	keepRecentTokens: number;
}

export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 16384,
	keepRecentTokens: 20000,
};

/**
 * The same settings, made to fit a window they were not written for.
 *
 * The defaults are absolute token counts chosen for a large context. On a small
 * one they collide: compaction fires at `contextWindow - reserve` and would then
 * keep more than the threshold that triggered it, so the cut could never get
 * back under the line. The trigger stays where upstream put it; only what a cut
 * keeps moves, and only downward. On a large window the ratio never binds.
 */
export function fitCompactionToWindow(
	settings: CompactionSettings,
	contextWindow: number | undefined,
): CompactionSettings {
	if (!contextWindow || contextWindow <= 0) return settings;
	const reserveTokens = Math.min(settings.reserveTokens, Math.floor(contextWindow / 2));
	const threshold = contextWindow - reserveTokens;
	// A quarter of the threshold: `findCutPoint` counts message text, while the
	// threshold counts the whole prompt, system prompt and tool schemas included,
	// so a larger keep budget can leave nothing to cut on a small window.
	const keepCeiling = Math.max(1024, Math.floor(threshold / 4));
	if (settings.keepRecentTokens <= keepCeiling && reserveTokens === settings.reserveTokens) {
		return settings;
	}
	return {
		...settings,
		reserveTokens,
		keepRecentTokens: Math.min(settings.keepRecentTokens, keepCeiling),
	};
}

// ============================================================================
// Token calculation
// ============================================================================

/**
 * Calculate total context tokens from usage.
 * Uses the native totalTokens field when available, falls back to computing from components.
 */
export function calculateContextTokens(usage: Usage): number {
	return usage.totalTokens || usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
}

/**
 * Get usage from an assistant message if available.
 * Skips aborted, error, and all-zero usage messages as they don't have valid usage data.
 */
function getAssistantUsage(msg: AgentMessage): Usage | undefined {
	if (msg.role === "assistant" && "usage" in msg) {
		const assistantMsg = msg as AssistantMessage;
		if (
			assistantMsg.stopReason !== "aborted" &&
			assistantMsg.stopReason !== "error" &&
			assistantMsg.usage &&
			calculateContextTokens(assistantMsg.usage) > 0
		) {
			return assistantMsg.usage;
		}
	}
	return undefined;
}

/**
 * Find the last valid assistant message usage from session entries.
 */
export function getLastAssistantUsage(entries: SessionEntry[]): Usage | undefined {
	for (let i = entries.length - 1; i >= 0; i--) {
		const entry = entries[i];
		if (entry.type === "message") {
			const usage = getAssistantUsage(entry.message);
			if (usage) return usage;
		}
	}
	return undefined;
}

export interface ContextUsageEstimate {
	tokens: number;
	usageTokens: number;
	trailingTokens: number;
	lastUsageIndex: number | null;
}

function getLastAssistantUsageInfo(messages: AgentMessage[]): { usage: Usage; index: number } | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const usage = getAssistantUsage(messages[i]);
		if (usage) return { usage, index: i };
	}
	return undefined;
}

/**
 * Estimate context tokens from messages, using the last assistant usage when available.
 * If there are messages after the last usage, estimate their tokens with estimateTokens.
 */
export function estimateContextTokens(messages: AgentMessage[]): ContextUsageEstimate {
	const usageInfo = getLastAssistantUsageInfo(messages);

	if (!usageInfo) {
		let estimated = 0;
		for (const message of messages) {
			estimated += estimateTokens(message);
		}
		return {
			tokens: estimated,
			usageTokens: 0,
			trailingTokens: estimated,
			lastUsageIndex: null,
		};
	}

	const usageTokens = calculateContextTokens(usageInfo.usage);
	let trailingTokens = 0;
	for (let i = usageInfo.index + 1; i < messages.length; i++) {
		trailingTokens += estimateTokens(messages[i]);
	}

	return {
		tokens: usageTokens + trailingTokens,
		usageTokens,
		trailingTokens,
		lastUsageIndex: usageInfo.index,
	};
}

/**
 * Check if compaction should trigger based on context usage.
 */
export function shouldCompact(contextTokens: number, contextWindow: number, settings: CompactionSettings): boolean {
	if (!settings.enabled) return false;
	const fitted = fitCompactionToWindow(settings, contextWindow);
	return contextTokens > contextWindow - fitted.reserveTokens;
}

// ============================================================================
// Cut point detection
// ============================================================================

const ESTIMATED_IMAGE_CHARS = 4800;

function estimateTextAndImageContentChars(content: string | Array<{ type: string; text?: string }>): number {
	if (typeof content === "string") {
		return content.length;
	}

	let chars = 0;
	for (const block of content) {
		if (block.type === "text" && block.text) {
			chars += block.text.length;
		} else if (block.type === "image") {
			chars += ESTIMATED_IMAGE_CHARS;
		}
	}
	return chars;
}

/**
 * Estimate token count for a message using chars/4 heuristic.
 * This is conservative (overestimates tokens).
 */
export function estimateTokens(message: AgentMessage): number {
	let chars = 0;

	switch (message.role) {
		case "user": {
			chars = estimateTextAndImageContentChars(
				(message as { content: string | Array<{ type: string; text?: string }> }).content,
			);
			return Math.ceil(chars / 4);
		}
		case "assistant": {
			const assistant = message as AssistantMessage;
			for (const block of assistant.content) {
				if (block.type === "text") {
					chars += block.text.length;
				} else if (block.type === "thinking") {
					chars += block.thinking.length;
				} else if (block.type === "toolCall") {
					chars += block.name.length + JSON.stringify(block.arguments).length;
				}
			}
			return Math.ceil(chars / 4);
		}
		case "custom":
		case "toolResult": {
			chars = estimateTextAndImageContentChars(message.content);
			return Math.ceil(chars / 4);
		}
		case "bashExecution": {
			chars = message.command.length + message.output.length;
			return Math.ceil(chars / 4);
		}
		case "branchSummary":
		case "compactionSummary": {
			chars = message.summary.length;
			return Math.ceil(chars / 4);
		}
	}

	return 0;
}

function isCutPointMessage(message: AgentMessage): boolean {
	switch (message.role) {
		case "user":
		case "assistant":
		case "bashExecution":
		case "custom":
		case "branchSummary":
		case "compactionSummary":
			return true;
		case "toolResult":
			return false;
	}
	return false;
}

function isTurnStartMessage(message: AgentMessage): boolean {
	switch (message.role) {
		case "user":
		case "bashExecution":
		case "custom":
		case "branchSummary":
		case "compactionSummary":
			return true;
		case "assistant":
		case "toolResult":
			return false;
	}
	return false;
}

function isTurnStartEntry(entry: SessionEntry): boolean {
	if (entry.type === "compaction") {
		return false;
	}
	return sessionEntryToContextMessages(entry).some(isTurnStartMessage);
}

/**
 * Find valid cut points: indices of context-visible user-like or assistant messages.
 * Never cut at tool results (they must follow their tool call).
 * When we cut at an assistant message with tool calls, its tool results follow it
 * and will be kept.
 */
function findValidCutPoints(entries: SessionEntry[], startIndex: number, endIndex: number): number[] {
	const cutPoints: number[] = [];
	for (let i = startIndex; i < endIndex; i++) {
		const entry = entries[i];
		if (entry.type === "compaction") {
			continue;
		}
		if (sessionEntryToContextMessages(entry).some(isCutPointMessage)) {
			cutPoints.push(i);
		}
	}
	return cutPoints;
}

/**
 * Find the context-visible user-role message that starts the turn containing the given entry index.
 * Returns -1 if no turn start found before the index.
 */
export function findTurnStartIndex(entries: SessionEntry[], entryIndex: number, startIndex: number): number {
	for (let i = entryIndex; i >= startIndex; i--) {
		if (isTurnStartEntry(entries[i])) {
			return i;
		}
	}
	return -1;
}

export interface CutPointResult {
	/** Index of first entry to keep */
	firstKeptEntryIndex: number;
	/** Index of user message that starts the turn being split, or -1 if not splitting */
	turnStartIndex: number;
	/** Whether this cut splits a turn (cut point is not a user message) */
	isSplitTurn: boolean;
}

/**
 * Find the cut point in session entries that keeps approximately `keepRecentTokens`.
 *
 * Algorithm: Walk backwards from newest, accumulating estimated message sizes.
 * Stop when we've accumulated >= keepRecentTokens. Cut at that point.
 *
 * Can cut at user OR assistant messages (never tool results). When cutting at an
 * assistant message with tool calls, its tool results come after and will be kept.
 *
 * Returns CutPointResult with:
 * - firstKeptEntryIndex: the entry index to start keeping from
 * - turnStartIndex: if cutting mid-turn, the user message that started that turn
 * - isSplitTurn: whether we're cutting in the middle of a turn
 *
 * Only considers entries between `startIndex` and `endIndex` (exclusive).
 */
export function findCutPoint(
	entries: SessionEntry[],
	startIndex: number,
	endIndex: number,
	keepRecentTokens: number,
): CutPointResult {
	const cutPoints = findValidCutPoints(entries, startIndex, endIndex);

	if (cutPoints.length === 0) {
		return { firstKeptEntryIndex: startIndex, turnStartIndex: -1, isSplitTurn: false };
	}

	// Walk backwards from newest, accumulating estimated message sizes
	let accumulatedTokens = 0;
	let cutIndex = cutPoints[0]; // Default: keep from first message (not header)

	for (let i = endIndex - 1; i >= startIndex; i--) {
		const entry = entries[i];
		const messageTokens = sessionEntryToContextMessages(entry).reduce(
			(sum, message) => sum + estimateTokens(message),
			0,
		);
		if (messageTokens === 0) continue;
		accumulatedTokens += messageTokens;

		// Check if we've exceeded the budget
		if (accumulatedTokens >= keepRecentTokens) {
			// The closest valid cut point at or after this entry -- and when there
			// is none, the last one there is.
			//
			// A cut point is never a tool result, so a keep budget spent by trailing
			// tool results alone has nothing at or after `i` to cut at. Falling back to
			// the first cut point would keep everything; the last one keeps the least.
			let found = false;
			for (let c = 0; c < cutPoints.length; c++) {
				if (cutPoints[c] >= i) {
					cutIndex = cutPoints[c];
					found = true;
					break;
				}
			}
			if (!found) cutIndex = cutPoints[cutPoints.length - 1];
			break;
		}
	}

	// Scan backwards from cutIndex to include adjacent metadata entries that do not affect context.
	while (cutIndex > startIndex) {
		const prevEntry = entries[cutIndex - 1];
		// Stop at compaction boundaries or context-visible entries.
		if (prevEntry.type === "compaction" || sessionEntryToContextMessages(prevEntry).length > 0) {
			break;
		}
		cutIndex--;
	}

	// Determine if this is a split turn
	const cutEntry = entries[cutIndex];
	const startsTurn = isTurnStartEntry(cutEntry);
	const turnStartIndex = startsTurn ? -1 : findTurnStartIndex(entries, cutIndex, startIndex);

	return {
		firstKeptEntryIndex: cutIndex,
		turnStartIndex,
		isSplitTurn: !startsTurn && turnStartIndex !== -1,
	};
}

// ============================================================================
// Summarization
// ============================================================================

const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

const UPDATE_SUMMARIZATION_INSTRUCTIONS = `Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

const UPDATE_SUMMARIZATION_PROMPT = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

${UPDATE_SUMMARIZATION_INSTRUCTIONS}`;

/**
 * Returns an error message when a summarization response cannot safely be persisted.
 * A length stop contains partial text and must not become a session checkpoint.
 */
export function getSummarizationFailure(response: AssistantMessage, label: string): string | undefined {
	if (response.stopReason === "error") {
		return `${label} failed: ${response.errorMessage || "Unknown error"}`;
	}
	if (response.stopReason === "length") {
		return `${label} failed: generation hit the token cap and the summary is incomplete`;
	}
	return undefined;
}

function createSummarizationOptions(
	model: Model<any>,
	maxTokens: number,
	apiKey: string | undefined,
	headers: Record<string, string> | undefined,
	env: Record<string, string> | undefined,
	signal: AbortSignal | undefined,
	thinkingLevel: ThinkingLevel | undefined,
	sessionId: string | undefined,
): SimpleStreamOptions {
	const options: SimpleStreamOptions = { maxTokens, signal, apiKey, headers, env, sessionId };
	if (model.reasoning && thinkingLevel && thinkingLevel !== "off") {
		options.reasoning = thinkingLevel;
	}
	return options;
}

/**
 * Shared choke point for every compaction/branch-summary summarization call. Wraps the
 * single LLM call in {@link retryAssistantCall} so transient stream drops (e.g.
 * `terminated`, socket close) honor the configured retry policy instead of failing
 * the whole compaction on the first attempt. Deterministic errors and aborts return
 * immediately (see {@link retryAssistantCall}).
 */
export async function completeSummarization(
	model: Model<any>,
	context: Context,
	options: SimpleStreamOptions,
	streamFn?: StreamFn,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
): Promise<AssistantMessage> {
	// Avoid cache writes for one-off summaries. Reuse caller-supplied routing when available;
	// callers without a session ID, including branch summaries, receive a fresh routing ID.
	const requestOptions: SimpleStreamOptions = {
		...options,
		cacheRetention: "none",
		sessionId: options.sessionId ?? uuidv7(),
	};
	const produce = async (): Promise<AssistantMessage> =>
		streamFn
			? (await streamFn(model, context, requestOptions)).result()
			: completeSimple(model, context, requestOptions);
	return retryAssistantCall(produce, retry, requestOptions.signal, callbacks);
}

/**
 * Generate a summary of the conversation using the LLM.
 * If previousSummary is provided, uses the update prompt to merge.
 */
export async function generateSummary(
	currentMessages: AgentMessage[],
	model: Model<any>,
	reserveTokens: number,
	apiKey: string | undefined,
	headers?: Record<string, string>,
	signal?: AbortSignal,
	customInstructions?: string,
	previousSummary?: string,
	thinkingLevel?: ThinkingLevel,
	streamFn?: StreamFn,
	env?: Record<string, string>,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
	sessionId?: string,
): Promise<string> {
	return (
		await generateSummaryWithUsage(
			currentMessages,
			model,
			reserveTokens,
			apiKey,
			headers,
			signal,
			customInstructions,
			previousSummary,
			thinkingLevel,
			streamFn,
			env,
			retry,
			callbacks,
			sessionId,
		)
	).text;
}

/** Build the provider context for a standalone summary request. */
function buildSummarizationContext(promptText: string): Context {
	return {
		systemPrompt: SUMMARIZATION_SYSTEM_PROMPT,
		messages: [
			{
				role: "user",
				content: [{ type: "text", text: promptText }],
				timestamp: Date.now(),
			},
		],
	};
}

/** Generate or update a conversation summary and return its provider usage. */

/**
 * How many tokens a summary may occupy.
 *
 * Bounded by the reserve and also by the window the summary has to live in: the
 * next compaction summarizes this one, so a summary sized for a large window
 * would grow every round on a small one until it filled the context. An eighth
 * leaves room for what compaction keeps and the work that follows.
 */
export function summaryTokenBudget(model: Model<any>, reserveTokens: number, share: number): number {
	const fromReserve = Math.floor(share * reserveTokens);
	const fromWindow = model.contextWindow > 0 ? Math.floor(model.contextWindow / 8) : Number.POSITIVE_INFINITY;
	const modelCeiling = model.maxTokens > 0 ? model.maxTokens : Number.POSITIVE_INFINITY;
	return Math.max(512, Math.min(fromReserve, fromWindow, modelCeiling));
}

const SUMMARIZATION_MARGIN_TOKENS = 512;
const MIN_SUMMARY_TOKENS = 512;

/**
 * Characters per token, for deciding whether a request fits.
 *
 * `estimateTokens` assumes 4, the usual figure for English prose; a coding
 * agent's context (code, logs, tool output) runs denser, around 2.7-4.0 on this
 * tokenizer. The figure sits below the typical case on purpose: fitting too
 * conservatively costs a shorter summary, fitting too loosely costs the whole
 * compaction.
 */
const CHARS_PER_TOKEN = 2.5;

/**
 * The session's first request, kept word for word through every compaction.
 *
 * A summary's "Goal" is a paraphrase, rewritten again by each later update. As
 * in Codex and hermes-agent, the user's own words are kept instead: for an agent
 * run on one task, the first request is the task, and its exact wording is what
 * the result is checked against.
 *
 * Only the first request, since a benchmark loop's notices also arrive as user
 * messages. It is re-read from the session each time, never from the previous
 * summary, so no update pass can paraphrase it.
 */
export const ORIGINAL_REQUEST_HEADING = "## Original request (verbatim)";
const ORIGINAL_REQUEST_END = "<!-- end of original request -->";
/** Codex's bound on user text kept through a compaction. */
const ORIGINAL_REQUEST_MAX_TOKENS = 20_000;

function userMessageText(message: AgentMessage): string | undefined {
	if (message.role !== "user") return undefined;
	const content = (message as { content?: unknown }).content;
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return undefined;
	const text = content
		.filter((block): block is { type: "text"; text: string } => block?.type === "text")
		.map((block) => block.text)
		.join("\n");
	return text || undefined;
}

/** The first user message on the path and where it sits, if there is one. */
function findOriginalRequest(entries: SessionEntry[]): { index: number; text: string } | undefined {
	for (let i = 0; i < entries.length; i++) {
		for (const message of sessionEntryToContextMessages(entries[i])) {
			const text = userMessageText(message);
			if (text?.trim()) return { index: i, text };
		}
	}
	return undefined;
}

/** A summary with the original request ahead of it, bounded by Codex's budget and the window. */
export function withOriginalRequest(summary: string, request: string, contextWindow?: number): string {
	let maxTokens = ORIGINAL_REQUEST_MAX_TOKENS;
	if (contextWindow && contextWindow > 0) maxTokens = Math.min(maxTokens, Math.floor(contextWindow / 16));
	const maxChars = Math.floor(maxTokens * CHARS_PER_TOKEN);
	const kept = request.length <= maxChars ? request : `${request.slice(0, maxChars)}\n[... request truncated ...]`;
	return `${ORIGINAL_REQUEST_HEADING}\n\n${kept}\n\n${ORIGINAL_REQUEST_END}\n\n${summary}`;
}

/** The summary without the verbatim request, so an update pass never sees it to rewrite. */
export function stripOriginalRequest(summary: string): string {
	if (!summary.startsWith(ORIGINAL_REQUEST_HEADING)) return summary;
	const end = summary.indexOf(ORIGINAL_REQUEST_END);
	return end === -1 ? summary : summary.slice(end + ORIGINAL_REQUEST_END.length).trimStart();
}

function estimatePromptTokens(text: string): number {
	return Math.ceil(text.length / CHARS_PER_TOKEN);
}
/** A length-stopped summary shorter than this is a fragment, not a checkpoint. */
const PARTIAL_SUMMARY_MIN_CHARS = 400;

/**
 * Keep a summarization request inside the window it has to fit in.
 *
 * The prompt (the history being dropped) and the completion (its summary) come
 * out of the same window. On a small window an unchecked request overflows, the
 * compaction fails, and the context keeps growing until it truncates. So the
 * completion is bounded by the room the prompt leaves it, and when the prompt
 * cannot leave enough, the prompt is what gives.
 */
export function fitSummarizationRequest(
	promptText: string,
	contextWindow: number,
	maxTokens: number,
): { promptText: string; maxTokens: number } {
	if (!contextWindow || contextWindow <= 0) return { promptText, maxTokens };
	const available = contextWindow - SUMMARIZATION_MARGIN_TOKENS;
	if (available <= MIN_SUMMARY_TOKENS) return { promptText, maxTokens };

	const promptTokens = estimatePromptTokens(promptText);
	if (promptTokens + maxTokens <= available) return { promptText, maxTokens };

	// The floor is not "enough to say something", it is enough to say something
	// *after thinking*: reasoning and the answer share one max_tokens on this
	// API, so a budget of a few hundred tokens is spent entirely on reasoning
	// and returns an empty summary, which is a failure rather than a short one.
	const floor = Math.min(maxTokens, Math.max(MIN_SUMMARY_TOKENS, Math.floor(contextWindow / 8)));
	const roomLeft = available - promptTokens;
	if (roomLeft >= floor) return { promptText, maxTokens: Math.min(maxTokens, roomLeft) };

	return {
		promptText: keepEnds(promptText, Math.floor((available - floor) * CHARS_PER_TOKEN)),
		maxTokens: floor,
	};
}

/**
 * Drop the middle. The head says what the task was and the tail says what was
 * just happening; a summary built from those is worth more than a request that
 * cannot be sent, which is the only other thing on offer at this size.
 */
function keepEnds(text: string, charLimit: number): string {
	if (charLimit <= 0 || text.length <= charLimit) return text;
	const marker = "\n\n[... middle of the conversation omitted to fit the summarization window ...]\n\n";
	const room = charLimit - marker.length;
	if (room <= 0) return text.slice(-charLimit);
	const head = Math.floor(room / 3);
	return text.slice(0, head) + marker + text.slice(text.length - (room - head));
}

/** A rejected request gets this many further attempts, each on half the prompt. */
const SUMMARIZATION_ATTEMPTS = 3;

/**
 * The same request with half the prompt.
 *
 * A character ratio cannot fit every request: characters per token is a property
 * of the text, from about 2 on dense output to 4 on prose. So the ratio is only
 * the opening guess, and a rejection is information -- halving the prompt
 * converges on something that fits without knowing the tokenizer.
 */
function halveSummarizationRequest(request: { promptText: string; maxTokens: number }): {
	promptText: string;
	maxTokens: number;
} {
	return { ...request, promptText: keepEnds(request.promptText, Math.floor(request.promptText.length / 2)) };
}

/**
 * Whether a smaller request could plausibly succeed where this one failed.
 *
 * Shrinking is only harmless when the failure was about size. pi already has a
 * retry policy for transient stream drops, and it deliberately does not retry
 * `insufficient_quota`, and does not retry at all when retry is disabled --
 * halving on *every* failure quietly retries those too, which three of its own
 * regression tests caught. So this asks the narrower question: did the request
 * not fit?
 *
 * A length stop did not fit by definition: the budget ran out mid-summary. The
 * other shape is the server refusing it outright, which on an OpenAI-compatible
 * endpoint is a 400 or 413 and often carries no body at all --
 * `Summarization failed: 400 status code (no body)` is the whole of what this
 * deployment says.
 */
function couldFitInSmaller(response: AssistantMessage): boolean {
	if (response.stopReason === "length") return true;
	if (response.stopReason !== "error") return false;
	const message = response.errorMessage ?? "";
	return /\b(?:400|413)\b/.test(message) || /context length|too large|too long|exceeds/i.test(message);
}

/**
 * The summary to persist, or a throw.
 *
 * A summary that stopped at the output cap still carries the work, so it is
 * kept rather than discarded; discarding it would leave the context where it
 * was, with no compaction at all.
 */
export function summaryText(response: AssistantMessage, label: string): string {
	const text = contentText(response.content);
	const failure = getSummarizationFailure(response, label);
	if (!failure) return text;
	if (response.stopReason === "length" && text.trim().length >= PARTIAL_SUMMARY_MIN_CHARS) {
		return `${text.trimEnd()}\n\n[Summary was cut off at the token cap; the sections above are complete.]`;
	}
	throw new Error(failure);
}

export async function generateSummaryWithUsage(
	currentMessages: AgentMessage[],
	model: Model<any>,
	reserveTokens: number,
	apiKey: string | undefined,
	headers?: Record<string, string>,
	signal?: AbortSignal,
	customInstructions?: string,
	previousSummary?: string,
	thinkingLevel?: ThinkingLevel,
	streamFn?: StreamFn,
	env?: Record<string, string>,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
	sessionId?: string,
): Promise<{ text: string; usage: Usage }> {
	const maxTokens = summaryTokenBudget(model, reserveTokens, 0.8);

	// Use update prompt if we have a previous summary, otherwise initial prompt
	let basePrompt = previousSummary ? UPDATE_SUMMARIZATION_PROMPT : SUMMARIZATION_PROMPT;
	if (customInstructions) {
		basePrompt = `${basePrompt}\n\nAdditional focus: ${customInstructions}`;
	}

	// Serialize conversation to text so model doesn't try to continue it
	// Convert to LLM messages first (handles custom types like bashExecution, custom, etc.)
	const llmMessages = convertToLlm(currentMessages);
	const conversationText = serializeConversation(llmMessages);

	// Build the prompt with conversation wrapped in tags
	let promptText = `<conversation>\n${conversationText}\n</conversation>\n\n`;
	if (previousSummary) {
		promptText += `<previous-summary>\n${previousSummary}\n</previous-summary>\n\n`;
	}
	promptText += basePrompt;

	let fitted = fitSummarizationRequest(promptText, model.contextWindow, maxTokens);

	for (let attempt = 1; ; attempt++) {
		const response = await completeSummarization(
			model,
			buildSummarizationContext(fitted.promptText),
			createSummarizationOptions(model, fitted.maxTokens, apiKey, headers, env, signal, thinkingLevel, sessionId),
			streamFn,
			retry,
			callbacks,
		);

		if (response.content.some((block) => block.type === "toolCall")) {
			throw new Error("Summarization attempted to call a tool");
		}

		try {
			return { text: summaryText(response, "Summarization"), usage: response.usage };
		} catch (error) {
			if (attempt >= SUMMARIZATION_ATTEMPTS || signal?.aborted || !couldFitInSmaller(response)) throw error;
			fitted = halveSummarizationRequest(fitted);
		}
	}
}

// ============================================================================
// Compaction Preparation (for extensions)
// ============================================================================

export interface CompactionPreparation {
	/** UUID of first entry to keep */
	firstKeptEntryId: string;
	/** Messages that will be summarized and discarded */
	messagesToSummarize: AgentMessage[];
	/** Messages that will be turned into turn prefix summary (if splitting) */
	turnPrefixMessages: AgentMessage[];
	/** Whether this is a split turn (cut point in middle of turn) */
	isSplitTurn: boolean;
	tokensBefore: number;
	/** Summary from previous compaction, for iterative update */
	previousSummary?: string;
	/** File operations extracted from messagesToSummarize */
	fileOps: FileOperations;
	/** The session's first request, when it is being compacted away; kept verbatim. */
	originalRequest?: string;
	/** Compaction settions from settings.jsonl	*/
	settings: CompactionSettings;
}

export function prepareCompaction(
	pathEntries: SessionEntry[],
	settings: CompactionSettings,
): CompactionPreparation | undefined {
	if (pathEntries.length > 0 && pathEntries[pathEntries.length - 1].type === "compaction") {
		return undefined;
	}

	let prevCompactionIndex = -1;
	for (let i = pathEntries.length - 1; i >= 0; i--) {
		if (pathEntries[i].type === "compaction") {
			prevCompactionIndex = i;
			break;
		}
	}

	let previousSummary: string | undefined;
	let boundaryStart = 0;
	if (prevCompactionIndex >= 0) {
		const prevCompaction = pathEntries[prevCompactionIndex] as CompactionEntry;
		previousSummary = stripOriginalRequest(prevCompaction.summary);
		const firstKeptEntryIndex = pathEntries.findIndex((entry) => entry.id === prevCompaction.firstKeptEntryId);
		boundaryStart = firstKeptEntryIndex >= 0 ? firstKeptEntryIndex : prevCompactionIndex + 1;
	}
	const boundaryEnd = pathEntries.length;

	const tokensBefore = estimateContextTokens(buildSessionContext(pathEntries).messages).tokens;

	const cutPoint = findCutPoint(pathEntries, boundaryStart, boundaryEnd, settings.keepRecentTokens);

	// Get UUID of first kept entry
	const firstKeptEntry = pathEntries[cutPoint.firstKeptEntryIndex];
	if (!firstKeptEntry?.id) {
		return undefined; // Session needs migration
	}
	const firstKeptEntryId = firstKeptEntry.id;

	const historyEnd = cutPoint.isSplitTurn ? cutPoint.turnStartIndex : cutPoint.firstKeptEntryIndex;

	// Messages to summarize (will be discarded after summary)
	const messagesToSummarize: AgentMessage[] = [];
	for (let i = boundaryStart; i < historyEnd; i++) {
		const msg = getMessageFromEntryForCompaction(pathEntries[i]);
		if (msg) messagesToSummarize.push(msg);
	}

	// Messages for turn prefix summary (if splitting a turn)
	const turnPrefixMessages: AgentMessage[] = [];
	if (cutPoint.isSplitTurn) {
		for (let i = cutPoint.turnStartIndex; i < cutPoint.firstKeptEntryIndex; i++) {
			const msg = getMessageFromEntryForCompaction(pathEntries[i]);
			if (msg) turnPrefixMessages.push(msg);
		}
	}

	if (messagesToSummarize.length === 0 && turnPrefixMessages.length === 0) {
		return undefined;
	}

	// Extract file operations from messages and previous compaction
	const fileOps = extractFileOperations(messagesToSummarize, pathEntries, prevCompactionIndex);

	// Also extract file ops from turn prefix if splitting
	if (cutPoint.isSplitTurn) {
		for (const msg of turnPrefixMessages) {
			extractFileOpsFromMessage(msg, fileOps);
		}
	}

	// Only when the cut takes it: a request still in the kept messages is already
	// there word for word.
	const firstRequest = findOriginalRequest(pathEntries);
	const originalRequest =
		firstRequest && firstRequest.index < cutPoint.firstKeptEntryIndex ? firstRequest.text : undefined;

	return {
		firstKeptEntryId,
		messagesToSummarize,
		turnPrefixMessages,
		isSplitTurn: cutPoint.isSplitTurn,
		tokensBefore,
		previousSummary,
		fileOps,
		settings,
		originalRequest,
	};
}

// ============================================================================
// Main compaction function
// ============================================================================

const TURN_PREFIX_SUMMARIZATION_PROMPT = `This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.`;

/**
 * Generate summaries for compaction using prepared data.
 * Returns CompactionResult - SessionManager adds uuid/parentUuid when saving.
 *
 * @param preparation - Pre-calculated preparation from prepareCompaction()
 * @param customInstructions - Optional custom focus for the summary
 * @param sessionId - Optional routing session ID forwarded without enabling prompt caching
 */
export async function compact(
	preparation: CompactionPreparation,
	model: Model<any>,
	apiKey: string | undefined,
	headers?: Record<string, string>,
	customInstructions?: string,
	signal?: AbortSignal,
	thinkingLevel?: ThinkingLevel,
	streamFn?: StreamFn,
	env?: Record<string, string>,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
	sessionId?: string,
): Promise<CompactionResult> {
	const {
		firstKeptEntryId,
		messagesToSummarize,
		turnPrefixMessages,
		isSplitTurn,
		tokensBefore,
		previousSummary,
		fileOps,
		settings,
	} = preparation;

	// Generate summaries and merge into one
	let summary: string;
	let summaryUsage: Usage;

	// A split turn with no history before it is the whole session so far -- one
	// request and everything done about it -- so it gets the full structured
	// summary rather than the short turn-prefix one, and carries any previous
	// summary forward.
	const prefixIsTheHistory = isSplitTurn && turnPrefixMessages.length > 0 && messagesToSummarize.length === 0;

	if (isSplitTurn && turnPrefixMessages.length > 0 && !prefixIsTheHistory) {
		let historyText = "No prior history.";
		let historyUsage: Usage | undefined;
		if (messagesToSummarize.length > 0) {
			const historyResult = await generateSummaryWithUsage(
				messagesToSummarize,
				model,
				settings.reserveTokens,
				apiKey,
				headers,
				signal,
				customInstructions,
				previousSummary,
				thinkingLevel,
				streamFn,
				env,
				retry,
				callbacks,
				sessionId,
			);
			historyText = historyResult.text;
			historyUsage = historyResult.usage;
		}
		const turnPrefixResult = await generateTurnPrefixSummary(
			turnPrefixMessages,
			model,
			settings.reserveTokens,
			apiKey,
			headers,
			env,
			signal,
			thinkingLevel,
			streamFn,
			retry,
			callbacks,
			sessionId,
		);
		// Merge into single summary
		summary = `${historyText}\n\n---\n\n**Turn Context (split turn):**\n\n${turnPrefixResult.text}`;
		summaryUsage = historyUsage ? combineUsage(historyUsage, turnPrefixResult.usage) : turnPrefixResult.usage;
	} else {
		// Just generate history summary
		const result = await generateSummaryWithUsage(
			prefixIsTheHistory ? turnPrefixMessages : messagesToSummarize,
			model,
			settings.reserveTokens,
			apiKey,
			headers,
			signal,
			customInstructions,
			previousSummary,
			thinkingLevel,
			streamFn,
			env,
			retry,
			callbacks,
			sessionId,
		);
		summary = result.text;
		summaryUsage = result.usage;
	}

	if (preparation.originalRequest) {
		summary = withOriginalRequest(summary, preparation.originalRequest, model.contextWindow);
	}

	// Compute file lists and append to summary
	const { readFiles, modifiedFiles } = computeFileLists(fileOps);
	summary += formatFileOperations(readFiles, modifiedFiles);

	if (!firstKeptEntryId) {
		throw new Error("First kept entry has no UUID - session may need migration");
	}

	return {
		summary,
		firstKeptEntryId,
		tokensBefore,
		usage: summaryUsage,
		details: { readFiles, modifiedFiles } as CompactionDetails,
	};
}

/**
 * Generate a summary for a turn prefix (when splitting a turn).
 */
async function generateTurnPrefixSummary(
	messages: AgentMessage[],
	model: Model<any>,
	reserveTokens: number,
	apiKey: string | undefined,
	headers?: Record<string, string>,
	env?: Record<string, string>,
	signal?: AbortSignal,
	thinkingLevel?: ThinkingLevel,
	streamFn?: StreamFn,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
	sessionId?: string,
): Promise<{ text: string; usage: Usage }> {
	const maxTokens = summaryTokenBudget(model, reserveTokens, 0.5); // Smaller budget for turn prefix
	const llmMessages = convertToLlm(messages);
	const conversationText = serializeConversation(llmMessages);
	const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${TURN_PREFIX_SUMMARIZATION_PROMPT}`;

	let fitted = fitSummarizationRequest(promptText, model.contextWindow, maxTokens);

	for (let attempt = 1; ; attempt++) {
		const response = await completeSummarization(
			model,
			buildSummarizationContext(fitted.promptText),
			createSummarizationOptions(model, fitted.maxTokens, apiKey, headers, env, signal, thinkingLevel, sessionId),
			streamFn,
			retry,
			callbacks,
		);

		if (response.content.some((block) => block.type === "toolCall")) {
			throw new Error("Turn prefix summarization attempted to call a tool");
		}

		try {
			return { text: summaryText(response, "Turn prefix summarization"), usage: response.usage };
		} catch (error) {
			if (attempt >= SUMMARIZATION_ATTEMPTS || signal?.aborted || !couldFitInSmaller(response)) throw error;
			fitted = halveSummarizationRequest(fitted);
		}
	}
}
