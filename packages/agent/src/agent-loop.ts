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
 * The loop has had no notion of time. A benchmark harness enforces one from
 * outside -- harbor kills the trial at its budget -- so the run ends mid-tool-
 * call with whatever was on disk at that instant, and the agent never knew it
 * was against a wall. Measured on Terminal-Bench 2.1: six of pi's twenty-four
 * failed trials ended that way, cut off with work in flight, while the median
 * successful trial used a fraction of the budget it was given. Neither pacing
 * error is available to fix from inside a loop that cannot see a clock.
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
 * Recovery for a turn cut off at the output token limit, in the two phases
 * Claude Code's loop uses (`max_output_tokens_escalate`, then
 * `max_output_tokens_recovery`).
 *
 * The order matters and is not obvious. A truncated turn usually means the
 * model needed more room, not that it did anything wrong, so the first
 * response is to **raise the ceiling and retry silently** -- no message, no
 * scolding. Only if it truncates again with the ceiling raised is the model
 * told, and that telling is bounded.
 *
 * Why pi needs it at all: the loop ends when an assistant message carries no
 * tool calls, which is right for a message that answers and wrong for one that
 * was cut off mid-sentence. The loop already knows `stopReason === "length"`
 * is dangerous -- when such a message *does* carry tool calls it refuses to run
 * them and asks for a reissue -- but with no tool calls there is nothing to
 * attach that notice to, so the guard never fires and the run ends on a
 * sentence that stops mid-word. Measured on Terminal-Bench 2.1 with a 27B
 * reasoning model: `regex-chess` spent 65,536 output tokens, 199,246
 * characters, entirely thinking, `stopReason: "length"`, and was recorded as
 * settled having taken no action at all.
 */
export const MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3;

/** What the model is told once the raised ceiling has also been used up. */
const OUTPUT_LIMIT_RECOVERY_NOTICE =
	"Output token limit hit. Resume directly from where you stopped -- do not " +
	"start over, and reach a tool call or a final answer before the limit.";

/**
 * The escalated ceiling for a retry, or undefined when there is no more room.
 *
 * The model's own `maxTokens` is the ceiling; escalating past it would only
 * produce a provider error. Escalation therefore happens at most once, and a
 * config already at the ceiling goes straight to the recovery phase.
 */
function escalatedMaxTokens(config: AgentLoopConfig): number | undefined {
	const ceiling = config.model.maxTokens;
	if (!ceiling || ceiling <= 0) return undefined;
	const current = config.maxTokens;
	if (current !== undefined && current >= ceiling) return undefined;
	return ceiling;
}

/**
 * The loop's recovery state, carried as one value rather than as loose flags.
 *
 * Claude Code's query loop threads a `State` object through every iteration and
 * rebuilds it immutably on each `continue` -- `maxOutputTokensRecoveryCount`,
 * `hasAttemptedReactiveCompact`, `maxOutputTokensOverride`, `turnCount`,
 * `transition` -- so that a later iteration can see which recoveries have
 * already been spent and refuse to repeat one.
 *
 * pi's loop had grown the same information as four unrelated locals: a boolean,
 * a counter, another boolean, and a per-iteration `let`. Three of them were
 * added by this work, each in the shape that suited the change being made, and
 * nothing outside the function could read any of them. Collecting them makes
 * the loop's recovery behaviour one value that can be logged, asserted on, and
 * extended without adding a fifth flag.
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
}

const INITIAL_LOOP_STATE: LoopState = {
	escalated: false,
	outputLimitRecoveries: 0,
	deadlineWarned: false,
	turnCount: 0,
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
			let transition: TurnTransition | undefined;
			// Scoped to a turn that ran *out* of ceiling, not one the provider
			// clamped below it. pi already recovers the second case a layer up:
			// `isRecoverableLength` is `usage.output < desiredMaxOutput`, and
			// AgentSession answers it by dropping the truncated message,
			// compacting, and retrying once -- the right move when the response
			// was cut short because context left no room. Acting on that case
			// here would take it away from the layer that handles it properly,
			// which is what three of its characterization tests caught.
			//
			// Nothing owned the other case: output that reached the ceiling
			// exactly. pi reads it as the model spending the budget it was given
			// and lets the run end, and with no tool call in the message the run
			// does end -- 36% of stock trials on this benchmark, whose ceiling
			// (16,384) sits inside the model's own output distribution (p99
			// 16,161).
			const ceiling = config.maxTokens ?? config.model.maxTokens;
			const spentTheCeiling =
				config.escalateOnSpentCeiling === true && ceiling > 0 && (message.usage?.output ?? 0) >= ceiling;
			if (toolCalls.length === 0 && message.stopReason === "length" && spentTheCeiling) {
				const ceiling = state.escalated ? undefined : escalatedMaxTokens(config);
				if (ceiling !== undefined) {
					state = { ...state, escalated: true };
					retrySilently = true;
					transition = { reason: "output_limit_escalate", maxTokens: ceiling };
					config = { ...config, maxTokens: ceiling };
				} else if (state.outputLimitRecoveries < MAX_OUTPUT_TOKENS_RECOVERY_LIMIT) {
					state = { ...state, outputLimitRecoveries: state.outputLimitRecoveries + 1 };
					recovery = OUTPUT_LIMIT_RECOVERY_NOTICE;
					transition = { reason: "output_limit_recovery", attempt: state.outputLimitRecoveries };
				}
			}
			if (toolCalls.length > 0) {
				// A "length" stop means the output was cut off by the token limit, so
				// every tool call in the message may carry truncated arguments. Fail
				// them all instead of executing potentially borked calls.
				const executedToolBatch =
					message.stopReason === "length"
						? await failToolCallsFromTruncatedMessage(toolCalls, emit)
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
			state = { ...state, turnCount: state.turnCount + 1 };
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
			pendingMessages = followUpMessages;
			continue;
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
): Promise<ExecutedToolCallBatch> {
	const messages: ToolResultMessage[] = [];
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
				`Tool call "${toolCall.name}" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.`,
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
