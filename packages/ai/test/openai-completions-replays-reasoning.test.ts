import { describe, expect, it } from "vitest";
import { convertMessages } from "../src/api/openai-completions.ts";
import type { AssistantMessage, Context, Model, OpenAICompletionsCompat, Usage } from "../src/types.ts";

/**
 * Reasoning is not conversation state for a plain OpenAI-compatible server. The
 * model produced it; what the next turn needs is the answer and the tool calls.
 * Sending it back only spends context.
 *
 * Measured over one 88-task run, by characters living in the replayed
 * conversation:
 *
 *     reasoning        12,233,431   39.7%
 *     assistant text    8,001,021   26.0%
 *     tool results      6,500,790   21.1%
 *     tool arguments    4,083,873   13.3%
 *
 * and on `regex-chess` it was 89% -- 507,799 characters of reasoning against
 * 61,746 of everything else. On a 32K window that is the whole problem: the
 * trials that failed sat at a median maximum context of 31,948 against 20,787
 * for the trials that solved.
 */

const emptyUsage: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const baseCompat = {
	supportsStore: true,
	supportsDeveloperRole: true,
	supportsReasoningEffort: true,
	supportsUsageInStreaming: true,
	supportsFinishReason: true,
	replaysReasoning: true,
	maxTokensField: "max_completion_tokens",
	requiresToolResultName: false,
	requiresStringToolResultContent: false,
	requiresThinkingAsText: false,
	requiresReasoningContentOnAssistantMessages: false,
	thinkingFormat: "openai",
	chatTemplateKwargs: {},
	chatTemplateArgs: {},
	supportsSystemMessages: true,
	supportsToolChoice: true,
	supportsParallelToolCalls: true,
	supportsStrictMode: true,
	supportsOpenAIGrammarTools: false,
	zaiToolStream: false,
	supportsPromptCaching: false,
	supportsLongCacheRetention: true,
} as unknown as Required<OpenAICompletionsCompat>;

function buildModel(): Model<"openai-completions"> {
	return {
		id: "m",
		name: "M",
		api: "openai-completions",
		provider: "p",
		baseUrl: "http://127.0.0.1:1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 32768,
		maxTokens: 8192,
		compat: baseCompat as never,
	};
}

function contextWithReasoning(): Context {
	const assistant: AssistantMessage = {
		role: "assistant",
		content: [
			{ type: "thinking", thinking: "a very long private deliberation", thinkingSignature: "reasoning_content" },
			{ type: "text", text: "the answer" },
		],
		api: "openai-completions",
		provider: "p",
		model: "m",
		usage: emptyUsage,
		stopReason: "stop",
		timestamp: 2,
	};
	return {
		systemPrompt: "",
		messages: [{ role: "user", content: "go", timestamp: 1 }, assistant],
		tools: [],
	} as unknown as Context;
}

describe("replaying reasoning back to the server", () => {
	it("sends it by default, which is the existing behaviour", () => {
		const messages = convertMessages(buildModel(), contextWithReasoning(), baseCompat as never);
		const assistant = messages[1] as unknown as Record<string, unknown>;
		expect(assistant.reasoning_content).toBe("a very long private deliberation");
	});

	it("omits it when the server has no use for it", () => {
		const compat = { ...baseCompat, replaysReasoning: false };
		const messages = convertMessages(buildModel(), contextWithReasoning(), compat as never);
		const assistant = messages[1] as unknown as Record<string, unknown>;
		expect(assistant.reasoning_content).toBeUndefined();
	});

	it("keeps the answer itself either way", () => {
		for (const replays of [true, false]) {
			const compat = { ...baseCompat, replaysReasoning: replays };
			const messages = convertMessages(buildModel(), contextWithReasoning(), compat as never);
			expect(JSON.stringify(messages[1])).toContain("the answer");
		}
	});
});
