/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	type Context,
	EventStream,
	type ToolResultMessage,
	validateToolArguments,
} from "@earendil-works/pi-ai";
import { getDefaultStreamFn } from "./stream-fn.ts";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	PrepareNextTurnContext,
	StreamFn,
	TurnTransition,
} from "./types.ts";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

/**
 * How long before the deadline the agent is told that time is running out.
 * One turn's worth: long enough to write down what it has, short enough that
 * it does not spend the rest of the run planning for the end.
 */
const DEADLINE_WARNING_MS = 5 * 60 * 1000;

/**
 * What the agent is told when the deadline is close, and when it has passed.
 *
 * A harness that enforces a time limit from outside would otherwise cut the run
 * off mid-tool-call; the warning gives the agent a turn to save its work and
 * write its answer.
 */
function deadlineNotice(remainingMs: number): string {
	if (remainingMs > 0) {
		const minutes = Math.max(1, Math.round(remainingMs / 60000));
		return (
			`You have about ${minutes} minute${minutes === 1 ? "" : "s"} left before this run is stopped. ` +
			"Finish what you can now: save work in progress, and if you have an answer, write it where the " +
			"task asked for it. Do not start anything you cannot complete in that time."
		);
	}
	return "Time is up. Stop and leave the work in its current state.";
}

/**
 * Cap on completion notices.
 *
 * The real brake is productivity: a notice whose round makes no tool call has
 * found an agent with nothing left to do, and asking again only spends turns.
 * The cap is a loose backstop. With no share configured the loop takes the
 * agent at its word, as in every interactive session.
 */
const DEFAULT_MAX_COMPLETION_NOTICES = 40;

function humanDuration(ms: number): string {
	const total = Math.max(0, Math.round(ms / 1000));
	const h = Math.floor(total / 3600);
	const m = Math.floor((total % 3600) / 60);
	if (h > 0) return `${h}h ${m}m`;
	if (m > 0) return `${m}m`;
	return `${total}s`;
}

/**
 * What the agent is told when it stops with most of its budget unspent.
 *
 * The loop treats "no tool calls" as done. This gives the agent what it cannot
 * see on its own -- how much of its run is left -- and asks for one pass against
 * the task's own wording rather than the checks it wrote for itself. The
 * threshold is self-limiting: each round it buys spends budget, so the share
 * climbs and the notices stop. The text quotes no share but the run's own, so
 * it holds under any budget.
 */
function budgetNotice(elapsedMs: number, budgetMs: number, steps: number, cutOff = false): string {
	const remaining = Math.max(0, budgetMs - elapsedMs);
	const pct = Math.round((elapsedMs / budgetMs) * 100);
	if (cutOff) {
		return (
			`Your last response was cut off at the output limit, so it never reached a tool call. You have been ` +
			`working ${humanDuration(elapsedMs)} over ${steps} steps, which is ${pct}% of your budget, and about ` +
			`${humanDuration(remaining)} is left.\n\n` +
			"Do not restate your reasoning. Pick up from where you stopped and make the next concrete move -- one " +
			"command, one edit, one file read -- so that this turn ends in an action rather than at the limit."
		);
	}
	return (
		`You have been working ${humanDuration(elapsedMs)} over ${steps} steps, which is ${pct}% of your ` +
		`budget. About ${humanDuration(remaining)} is left.\n\n` +
		"Measured on this benchmark: about half of the runs that failed were not out of time -- they stopped " +
		"on their own with time left, on a check the agent had written for itself and then passed.\n\n" +
		"If time is what you have left, the cheapest thing you can do with it is one pass you have not done: " +
		"re-read the task's own words, list what will be run against your work, and check the deliverable " +
		"against that list rather than against the checks you already wrote. If you have genuinely finished " +
		"and verified against the task's own wording, say so and stop."
	);
}

/**
 * Recovery for a turn cut off at the output token limit, in two phases (the
 * order Claude Code uses): raise the ceiling and retry silently first, since a
 * truncated turn usually means the model needed more room; only if it
 * truncates again is the model told, a bounded number of times.
 *
 * Without it, a turn cut off mid-sentence with no tool call would be read as
 * the agent finishing.
 */
/**
 * Slack left between the escalated ceiling and the context window. The input
 * count is the previous turn's, and the recovery notice this run is about to
 * add is not in it.
 */
const CONTEXT_SAFETY_MARGIN_TOKENS = 512;

export const MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3;

/** Consecutive budget notices a truncated round may buy without any tool call. */
export const MAX_TRUNCATION_NOTICES = 3;

/** What the model is told once the raised ceiling has also been used up. */
/**
 * Claude Code's recovery text. The last clause matters: resuming is advice about
 * the turn that was cut, breaking the work up is advice about the next one,
 * which would otherwise hit the same ceiling.
 */
const OUTPUT_LIMIT_RECOVERY_NOTICE =
	"Output token limit hit. Resume directly -- no apology, no recap of what you were doing. " +
	"Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.";

/** Most of a cut-off reasoning block that is quoted back, in characters. */
export const CARRIED_REASONING_MAX_CHARS = 8000;

/** Characters per token, for bounding the quote by the window. */
const CARRIED_REASONING_CHARS_PER_TOKEN = 2.5;

/**
 * The end of a turn's reasoning, when reasoning is all it produced, so the
 * recovery can hand it back.
 *
 * "Pick up mid-thought" assumes the thought survives. Where reasoning and answer
 * share one `max_tokens`, a cut (or a stop) inside the reasoning leaves nothing
 * the next request carries: a message with no text and no tool call is dropped,
 * and chat templates strip earlier reasoning anyway. Without the quote the model
 * re-derives its plan from scratch and tends to hit the same limit.
 *
 * The end is quoted rather than the start -- the start is the plan, the end is
 * the work -- bounded to a sixteenth of the window.
 */
export function strandedReasoning(message: AssistantMessage, contextWindow?: number): string | undefined {
	const parts: string[] = [];
	for (const block of message.content) {
		if (block.type === "toolCall") return undefined;
		if (block.type === "text" && block.text.trim()) return undefined;
		if (block.type === "thinking" && !block.redacted && block.thinking.trim()) parts.push(block.thinking);
	}
	const reasoning = parts.join("\n").trim();
	if (!reasoning) return undefined;
	let max = CARRIED_REASONING_MAX_CHARS;
	if (contextWindow && contextWindow > 0) {
		max = Math.min(max, Math.floor((contextWindow / 16) * CARRIED_REASONING_CHARS_PER_TOKEN));
	}
	if (reasoning.length <= max) return reasoning;
	let tail = reasoning.slice(-max);
	// Open on a line boundary when one is near, not mid-word.
	const newline = tail.indexOf("\n");
	if (newline >= 0 && newline < max / 5) tail = tail.slice(newline + 1);
	return `...${tail}`;
}

/** Whether a turn said nothing and did nothing: no text, no tool call. */
export function isEmptyAnswer(message: AssistantMessage): boolean {
	return !message.content.some(
		(block) => block.type === "toolCall" || (block.type === "text" && block.text.trim().length > 0),
	);
}

export const MAX_EMPTY_ANSWER_RECOVERIES = 2;

/**
 * What the model is told when a turn stopped inside its reasoning, with neither
 * an answer nor a tool call. That is a stall, not a completion (hermes-agent
 * treats it the same way), so the model gets its reasoning back and is asked to
 * go on -- at most twice in a row -- or to say in words that it is finished.
 */
const EMPTY_ANSWER_NOTICE =
	"Your last turn ended inside your reasoning, with no answer and no tool call, so nothing was done and " +
	"nothing was said. Continue from where the reasoning stopped and make the next concrete move. If the task " +
	"is already finished, say so in words.";

/** A recovery notice with the stranded reasoning quoted ahead of it, when there is any. */
function withCarriedReasoning(notice: string, reasoning: string | undefined): string {
	if (!reasoning) return notice;
	return (
		"Your reasoning is not kept between turns, and this turn ended before it reached any text or tool call, " +
		"so here is where it stopped, quoted from its end:\n\n" +
		`<previous_reasoning>\n${reasoning}\n</previous_reasoning>\n\n` +
		notice
	);
}

/**
 * The escalated ceiling for a retry, or undefined when there is no more room.
 *
 * The model's own `maxTokens` is the ceiling; escalating past it would only
 * produce a provider error. Escalation therefore happens at most once, and a
 * config already at the ceiling goes straight to the recovery phase.
 */
function escalatedMaxTokens(config: AgentLoopConfig, inputTokens?: number): number | undefined {
	const ceiling = config.model.maxTokens;
	if (!ceiling || ceiling <= 0) return undefined;
	const current = config.maxTokens;
	if (current !== undefined && current >= ceiling) return undefined;
	// The raised ceiling has to fit beside the context already in the window, not
	// just be a number the model accepts; otherwise the retry becomes a rejected
	// request, which is worse than the truncation it was meant to recover.
	const window = config.model.contextWindow;
	if (window && window > 0 && inputTokens !== undefined && inputTokens > 0) {
		const room = window - inputTokens - CONTEXT_SAFETY_MARGIN_TOKENS;
		if (room <= (current ?? 0)) return undefined;
		return Math.min(ceiling, room);
	}
	return ceiling;
}

/**
 * The raised ceiling a turn cut off at its output limit earns, or undefined.
 *
 * Once per run, only when the setting asks for it, and only for a turn that
 * spent the whole ceiling -- one cut short below it is pi's own case, handled a
 * layer up. The same terms whether or not the turn reached a tool call.
 */
function ceilingAfterCut(message: AssistantMessage, config: AgentLoopConfig, state: LoopState): number | undefined {
	if (state.escalated || config.escalateOnSpentCeiling !== true) return undefined;
	if ((message.usage?.output ?? 0) < (config.maxTokens ?? 0)) return undefined;
	return escalatedMaxTokens(config, message.usage?.input);
}

/**
 * The loop's recovery state, carried as one immutable value rather than loose
 * flags (the shape of Claude Code's query loop), so a later iteration can see
 * which recoveries have been spent, and the state can be logged and asserted on.
 */
interface LoopState {
	/** The output ceiling has been raised once; a second truncation speaks. */
	readonly escalated: boolean;
	/** Recovery notices sent, bounded by MAX_OUTPUT_TOKENS_RECOVERY_LIMIT. */
	readonly outputLimitRecoveries: number;
	/** The deadline warning is sent once; repeated, it becomes the subject. */
	readonly deadlineWarned: boolean;
	/** Turns completed, counting every assistant response the loop has taken. */
	readonly turnCount: number;
	/** Completion notices sent, bounded by `maxCompletionNotices`. */
	readonly completionNotices: number;
	/** Tool calls the loop has run, to tell a bought round from a spent one. */
	readonly toolCallsMade: number;
	/** `toolCallsMade` when the last completion notice was sent. */
	readonly toolCallsAtLastNotice: number;
	/** Consecutive notices bought by truncation alone, bounded by MAX_TRUNCATION_NOTICES. */
	readonly truncationNotices: number;
	/** Consecutive empty answers nudged, bounded by MAX_EMPTY_ANSWER_RECOVERIES. */
	readonly emptyAnswerRecoveries: number;
}

const INITIAL_LOOP_STATE: LoopState = {
	escalated: false,
	outputLimitRecoveries: 0,
	deadlineWarned: false,
	turnCount: 0,
	completionNotices: 0,
	toolCallsMade: 0,
	toolCallsAtLastNotice: -1,
	truncationNotices: 0,
	emptyAnswerRecoveries: 0,
};

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	void runAgentLoop(
		prompts,
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	void runAgentLoopContinue(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): Promise<AgentMessage[]> {
	const newMessages: AgentMessage[] = [...prompts];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...prompts],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const prompt of prompts) {
		await emit({ type: "message_start", message: prompt });
		await emit({ type: "message_end", message: prompt });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn ?? getDefaultStreamFn());
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn ?? getDefaultStreamFn());
	return newMessages;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
	initialContext: AgentContext,
	newMessages: AgentMessage[],
	initialConfig: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFunction: StreamFn,
): Promise<void> {
	let currentContext = initialContext;
	let config = initialConfig;
	let lastCompletedTurn: PrepareNextTurnContext | undefined;
	// Consecutive turns handed back for producing neither a tool call nor an
	// answer. Reset by any turn that does one of those.
	// Recovery state as one value; see `LoopState`. Replaced immutably at each
	// point that spends a recovery, so what has been tried is always readable
	// as a whole rather than reconstructed from four separate flags.
	let state: LoopState = INITIAL_LOOP_STATE;
	// Set when the loop continues past a point where it would have stopped;
	// carried to the next turn's `turn_end`, which is where that decision shows.
	let pendingTransition: TurnTransition | undefined;
	// Check for steering messages at start (user may have typed while waiting)
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	while (true) {
		let hasMoreToolCalls = true;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			if (lastCompletedTurn) {
				const nextTurnSnapshot = await config.prepareNextTurn?.(lastCompletedTurn);
				if (nextTurnSnapshot) {
					currentContext = nextTurnSnapshot.context ?? currentContext;
					config = {
						...config,
						model: nextTurnSnapshot.model ?? config.model,
						reasoning:
							nextTurnSnapshot.thinkingLevel === undefined
								? config.reasoning
								: nextTurnSnapshot.thinkingLevel === "off"
									? undefined
									: nextTurnSnapshot.thinkingLevel,
					};
				}
				// Preparation can be long-running (for example, compaction). Pick up steering
				// queued while it ran. Only poll again if the earlier poll returned nothing;
				// otherwise one-at-a-time mode would deliver two messages in this turn.
				if (pendingMessages.length === 0) {
					pendingMessages = (await config.getSteeringMessages?.()) || [];
				}
				await emit({ type: "turn_start" });
			}

			// Process pending messages (inject before next assistant response)
			if (pendingMessages.length > 0) {
				for (const message of pendingMessages) {
					await emit({ type: "message_start", message });
					await emit({ type: "message_end", message });
					currentContext.messages.push(message);
					newMessages.push(message);
				}
				pendingMessages = [];
			}

			// Stop on our own terms rather than being killed mid-turn.
			if (config.deadline !== undefined) {
				const remaining = config.deadline - Date.now();
				if (remaining <= 0) {
					await emit({ type: "agent_end", messages: newMessages });
					return;
				}
				if (!state.deadlineWarned && remaining <= DEADLINE_WARNING_MS) {
					state = { ...state, deadlineWarned: true };
					const notice: AgentMessage = {
						role: "user",
						content: [{ type: "text", text: deadlineNotice(remaining) }],
						timestamp: Date.now(),
					};
					await emit({ type: "message_start", message: notice });
					await emit({ type: "message_end", message: notice });
					currentContext.messages.push(notice);
					newMessages.push(notice);
				}
			}

			// Stream assistant response
			const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFunction);
			newMessages.push(message);

			if (message.stopReason === "error" || message.stopReason === "aborted") {
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			// Check for tool calls
			const toolCalls = message.content.filter((c) => c.type === "toolCall");

			const toolResults: ToolResultMessage[] = [];
			hasMoreToolCalls = false;
			// A turn with no tool call usually means the agent is done. One kind
			// looks like that and is not: a turn cut off at the output limit,
			// which stopped mid-sentence. Raise the ceiling and retry silently
			// first; only tell the model once the raised ceiling is used up too.
			let recovery: string | undefined;
			let retrySilently = false;
			let transition: TurnTransition | undefined = pendingTransition;
			pendingTransition = undefined;
			// The recovery always applies: a turn that stopped on `length` with no tool
			// call produced nothing usable. Raising the ceiling sits behind
			// `escalateOnSpentCeiling`, since it changes what the model is asked for.
			if (toolCalls.length === 0 && message.stopReason === "length") {
				// The escalation half stays behind the setting: raising the
				// ceiling changes what the model is asked for, and pi has a
				// characterization test pinning the old behaviour. The recovery
				// half does not, per the comment above.
				const ceiling = ceilingAfterCut(message, config, state);
				if (ceiling !== undefined) {
					state = { ...state, escalated: true };
					retrySilently = true;
					transition = { reason: "output_limit_escalate", maxTokens: ceiling };
					config = { ...config, maxTokens: ceiling };
				} else if (state.outputLimitRecoveries < MAX_OUTPUT_TOKENS_RECOVERY_LIMIT) {
					state = { ...state, outputLimitRecoveries: state.outputLimitRecoveries + 1 };
					recovery = withCarriedReasoning(
						OUTPUT_LIMIT_RECOVERY_NOTICE,
						strandedReasoning(message, config.model.contextWindow),
					);
					transition = { reason: "output_limit_recovery", attempt: state.outputLimitRecoveries };
				}
			}
			// Stopped (not cut off) inside its reasoning: a stall rather than an answer.
			// Only with reasoning to hand back; a turn with nothing at all still ends the
			// run as before.
			const stranded =
				message.stopReason === "stop" && state.emptyAnswerRecoveries < MAX_EMPTY_ANSWER_RECOVERIES
					? strandedReasoning(message, config.model.contextWindow)
					: undefined;
			if (stranded !== undefined) {
				state = { ...state, emptyAnswerRecoveries: state.emptyAnswerRecoveries + 1 };
				recovery = withCarriedReasoning(EMPTY_ANSWER_NOTICE, stranded);
				transition = { reason: "empty_answer_recovery", attempt: state.emptyAnswerRecoveries };
			}
			if (toolCalls.length > 0) {
				// A "length" stop means the output was cut off by the token limit, so
				// every tool call in the message may carry truncated arguments. Fail
				// them all instead of executing potentially borked calls.
				//
				// The ceiling is raised on the same terms as a cut turn with no tool call,
				// as Claude Code and hermes-agent do, so the reissued call fits.
				let raised = false;
				if (message.stopReason === "length") {
					const ceiling = ceilingAfterCut(message, config, state);
					if (ceiling !== undefined) {
						state = { ...state, escalated: true };
						transition = { reason: "output_limit_escalate", maxTokens: ceiling };
						config = { ...config, maxTokens: ceiling };
						raised = true;
					}
				}
				const executedToolBatch =
					message.stopReason === "length"
						? await failToolCallsFromTruncatedMessage(toolCalls, emit, raised)
						: await executeToolCalls(currentContext, message, config, signal, emit);
				toolResults.push(...executedToolBatch.messages);
				hasMoreToolCalls = !executedToolBatch.terminate;

				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
				}
			}

			if (!transition && toolCalls.length > 0 && hasMoreToolCalls) {
				transition = { reason: "tool_calls", count: toolCalls.length };
			}
			state = {
				...state,
				turnCount: state.turnCount + 1,
				toolCallsMade: state.toolCallsMade + toolCalls.length,
				// Consecutive: any turn that said or did something clears it.
				emptyAnswerRecoveries: isEmptyAnswer(message) ? state.emptyAnswerRecoveries : 0,
			};
			await emit({ type: "turn_end", message, toolResults, transition });

			lastCompletedTurn = {
				message,
				toolResults,
				context: currentContext,
				newMessages,
			};

			if (await config.shouldStopAfterTurn?.(lastCompletedTurn)) {
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			pendingMessages = (await config.getSteeringMessages?.()) || [];
			if (!transition && pendingMessages.length > 0) {
				transition = { reason: "steering" };
			}
			// The escalation phase adds nothing to the conversation -- the point is
			// that the model gets more room, not that it is told off for needing it.
			if (retrySilently) {
				hasMoreToolCalls = true;
			}
			// Appended after the steering poll, which reassigns the array. Delivered
			// as a user message because the truncated-tool-call path this mirrors
			// speaks through a tool result, and here there is no tool call to answer.
			if (recovery) {
				pendingMessages.push({
					role: "user",
					content: [{ type: "text", text: recovery }],
					timestamp: Date.now(),
				});
			}
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			// Set as pending so inner loop processes them
			pendingTransition = { reason: "follow_up", count: followUpMessages.length };
			pendingMessages = followUpMessages;
			continue;
		}

		// The agent says it is done. Ask again while most of the budget is unspent
		// -- see `budgetNotice` for what stopping early costs on this benchmark.
		const budgetMs = config.timeBudgetMs;
		const share = config.stopBudgetShare;
		if (budgetMs !== undefined && budgetMs > 0 && share !== undefined && config.deadline !== undefined) {
			const elapsed = budgetMs - (config.deadline - Date.now());
			const used = elapsed / budgetMs;
			const cap = config.maxCompletionNotices ?? DEFAULT_MAX_COMPLETION_NOTICES;
			// A notice whose round ran no tool call bought nothing; a second one
			// buys nothing either, and the turns come out of the same budget.
			//
			// Unless the round was cut off: a turn that stopped on `length` did not
			// decline to act, it was truncated, so it still earns a notice.
			const cutOff = lastCompletedTurn?.message.stopReason === "length";
			const bought = state.toolCallsAtLastNotice < 0 || state.toolCallsMade > state.toolCallsAtLastNotice;
			// A truncated round earns the next notice, but not forever: a run that keeps
			// truncating would otherwise collect every notice and spend them producing
			// nothing. Truncation re-arms the notice at most MAX_TRUNCATION_NOTICES
			// times in a row; a round that calls a tool clears the count.
			const truncationBudget = cutOff && state.truncationNotices < MAX_TRUNCATION_NOTICES;
			if (used >= 0 && used < share && state.completionNotices < cap && (bought || truncationBudget)) {
				state = {
					...state,
					completionNotices: state.completionNotices + 1,
					toolCallsAtLastNotice: state.toolCallsMade,
					truncationNotices: bought ? 0 : state.truncationNotices + 1,
				};
				pendingTransition = { reason: "budget_notice", attempt: state.completionNotices, shareUsed: used };
				pendingMessages = [
					{
						role: "user",
						content: [
							{
								type: "text",
								text: withCarriedReasoning(
									budgetNotice(elapsed, budgetMs, state.turnCount, cutOff),
									lastCompletedTurn
										? strandedReasoning(lastCompletedTurn.message, config.model.contextWindow)
										: undefined,
								),
							},
						],
						timestamp: Date.now(),
					},
				];
				continue;
			}
		}

		// No more messages, exit
		break;
	}

	await emit({ type: "agent_end", messages: newMessages });
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 */
async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFunction: StreamFn,
): Promise<AssistantMessage> {
	// Apply context transform if configured (AgentMessage[] → AgentMessage[])
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	// Convert to LLM-compatible messages (AgentMessage[] → Message[])
	const llmMessages = await config.convertToLlm(messages);

	// Build LLM context
	const llmContext: Context = {
		systemPrompt: context.systemPrompt,
		messages: llmMessages,
		tools: context.tools,
	};

	// Resolve API key (important for expiring tokens)
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;

	const response = await streamFunction(config.model, llmContext, {
		...config,
		apiKey: resolvedApiKey,
		signal,
	});

	let partialMessage: AssistantMessage | null = null;
	let addedPartial = false;

	for await (const event of response) {
		switch (event.type) {
			case "start":
				partialMessage = event.partial;
				context.messages.push(partialMessage);
				addedPartial = true;
				await emit({ type: "message_start", message: { ...partialMessage } });
				break;

			case "text_start":
			case "text_delta":
			case "text_end":
			case "thinking_start":
			case "thinking_delta":
			case "thinking_end":
			case "toolcall_start":
			case "toolcall_delta":
			case "toolcall_end":
				if (partialMessage) {
					partialMessage = event.partial;
					context.messages[context.messages.length - 1] = partialMessage;
					await emit({
						type: "message_update",
						assistantMessageEvent: event,
						message: { ...partialMessage },
					});
				}
				break;

			case "done":
			case "error": {
				const finalMessage = await response.result();
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
				}
				if (!addedPartial) {
					await emit({ type: "message_start", message: { ...finalMessage } });
				}
				await emit({ type: "message_end", message: finalMessage });
				return finalMessage;
			}
		}
	}

	const finalMessage = await response.result();
	if (addedPartial) {
		context.messages[context.messages.length - 1] = finalMessage;
	} else {
		context.messages.push(finalMessage);
		await emit({ type: "message_start", message: { ...finalMessage } });
	}
	await emit({ type: "message_end", message: finalMessage });
	return finalMessage;
}

/**
 * Fail all tool calls from an assistant message that was truncated by the
 * output token limit. Streamed tool-call arguments are finalized with a
 * best-effort JSON salvage parser, so a truncated message can yield tool calls
 * whose arguments parse and validate but are silently incomplete. None of them
 * are safe to execute; report each as an error so the model can re-issue them.
 */
async function failToolCallsFromTruncatedMessage(
	toolCalls: AgentToolCall[],
	emit: AgentEventSink,
	raised = false,
): Promise<ExecutedToolCallBatch> {
	const messages: ToolResultMessage[] = [];
	// With room made, the call can come back whole. Without it, the same call
	// hits the same limit, so the only way through is Claude Code's advice:
	// smaller pieces.
	const next = raised
		? "The limit has been raised for your next response: re-issue the tool call with complete arguments."
		: "Re-issue the tool call with complete arguments, and if it is too large for one response, split it: write the file in parts, or make several smaller edits.";
	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});
		const finalized: FinalizedToolCallOutcome = {
			toolCall,
			result: createErrorToolResult(
				`Tool call "${toolCall.name}" was not executed: the response hit the output token limit, so its arguments may be truncated. ${next}`,
			),
			isError: true,
		};
		await emitToolExecutionEnd(finalized, emit);
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		messages.push(toolResultMessage);
	}
	return { messages, terminate: false };
}

/**
 * Execute tool calls from an assistant message.
 */
async function executeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	const hasSequentialToolCall = toolCalls.some(
		(tc) => currentContext.tools?.find((t) => t.name === tc.name)?.executionMode === "sequential",
	);
	if (config.toolExecution === "sequential" || hasSequentialToolCall) {
		return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}

type ExecutedToolCallBatch = {
	messages: ToolResultMessage[];
	terminate: boolean;
};

async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallOutcome[] = [];
	const messages: ToolResultMessage[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		let finalized: FinalizedToolCallOutcome;
		if (preparation.kind === "immediate") {
			finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			};
		} else {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
		}

		await emitToolExecutionEnd(finalized, emit);
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		finalizedCalls.push(finalized);
		messages.push(toolResultMessage);

		if (signal?.aborted) {
			break;
		}
	}

	return {
		messages,
		terminate: shouldTerminateToolBatch(finalizedCalls),
	};
}

async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallEntry[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			const finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			} satisfies FinalizedToolCallOutcome;
			await emitToolExecutionEnd(finalized, emit);
			finalizedCalls.push(finalized);
			if (signal?.aborted) {
				break;
			}
			continue;
		}

		finalizedCalls.push(async () => {
			if (signal?.aborted) {
				const finalized = {
					toolCall,
					result: createErrorToolResult("Operation aborted"),
					isError: true,
				} satisfies FinalizedToolCallOutcome;
				await emitToolExecutionEnd(finalized, emit);
				return finalized;
			}
			const executed = await executePreparedToolCall(preparation, signal, emit);
			const finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
			await emitToolExecutionEnd(finalized, emit);
			return finalized;
		});
		if (signal?.aborted) {
			break;
		}
	}

	const orderedFinalizedCalls = await Promise.all(
		finalizedCalls.map((entry) => (typeof entry === "function" ? entry() : Promise.resolve(entry))),
	);
	const messages: ToolResultMessage[] = [];
	for (const finalized of orderedFinalizedCalls) {
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		messages.push(toolResultMessage);
	}

	return {
		messages,
		terminate: shouldTerminateToolBatch(orderedFinalizedCalls),
	};
}

type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool<any>;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<any>;
	isError: boolean;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallOutcome = {
	toolCall: AgentToolCall;
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallEntry = FinalizedToolCallOutcome | (() => Promise<FinalizedToolCallOutcome>);

function shouldTerminateToolBatch(finalizedCalls: FinalizedToolCallOutcome[]): boolean {
	return finalizedCalls.length > 0 && finalizedCalls.every((finalized) => finalized.result.terminate === true);
}

function prepareToolCallArguments(tool: AgentTool<any>, toolCall: AgentToolCall): AgentToolCall {
	if (!tool.prepareArguments) {
		return toolCall;
	}
	const preparedArguments = tool.prepareArguments(toolCall.arguments);
	if (preparedArguments === toolCall.arguments) {
		return toolCall;
	}
	return {
		...toolCall,
		arguments: preparedArguments as Record<string, any>,
	};
}

async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
		};
	}

	try {
		const preparedToolCall = prepareToolCallArguments(tool, toolCall);
		const validatedArgs = validateToolArguments(tool, preparedToolCall);
		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(
				{
					assistantMessage,
					toolCall,
					args: validatedArgs,
					context: currentContext,
				},
				signal,
			);
			if (signal?.aborted) {
				return {
					kind: "immediate",
					result: createErrorToolResult("Operation aborted"),
					isError: true,
				};
			}
			if (beforeResult?.block) {
				const result = createErrorToolResult(beforeResult.reason || "Tool execution was blocked");
				if (beforeResult.terminate === true) {
					result.terminate = true;
				}
				return {
					kind: "immediate",
					result,
					isError: true,
				};
			}
		}
		if (signal?.aborted) {
			return {
				kind: "immediate",
				result: createErrorToolResult("Operation aborted"),
				isError: true,
			};
		}
		return {
			kind: "prepared",
			toolCall,
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];
	let acceptingUpdates = true;

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				if (!acceptingUpdates) return;
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		acceptingUpdates = false;
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		acceptingUpdates = false;
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	} finally {
		acceptingUpdates = false;
	}
}

async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<FinalizedToolCallOutcome> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		try {
			const afterResult = await config.afterToolCall(
				{
					assistantMessage,
					toolCall: prepared.toolCall,
					args: prepared.args,
					result,
					isError,
					context: currentContext,
				},
				signal,
			);
			if (afterResult) {
				result = {
					...result,
					content: afterResult.content ?? result.content,
					details: afterResult.details ?? result.details,
					usage: afterResult.usage ?? result.usage,
					terminate: afterResult.terminate ?? result.terminate,
				};
				isError = afterResult.isError ?? isError;
			}
		} catch (error) {
			result = createErrorToolResult(error instanceof Error ? error.message : String(error));
			isError = true;
		}
	}

	return {
		toolCall: prepared.toolCall,
		result,
		isError,
	};
}

function createErrorToolResult(message: string): AgentToolResult<any> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolExecutionEnd(finalized: FinalizedToolCallOutcome, emit: AgentEventSink): Promise<void> {
	await emit({
		type: "tool_execution_end",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		result: finalized.result,
		isError: finalized.isError,
	});
}

function createToolResultMessage(finalized: FinalizedToolCallOutcome): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		// Untyped tools (JS extensions) can return results without content; normalize
		// so the null never enters session history or provider payloads.
		content: finalized.result.content ?? [],
		details: finalized.result.details,
		usage: finalized.result.usage,
		...(finalized.result.addedToolNames?.length ? { addedToolNames: finalized.result.addedToolNames } : {}),
		isError: finalized.isError,
		timestamp: Date.now(),
	};
}

async function emitToolResultMessage(toolResultMessage: ToolResultMessage, emit: AgentEventSink): Promise<void> {
	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
}
