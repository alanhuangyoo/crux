import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Context, Model } from "@earendil-works/pi-ai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
	type CompactionPreparation,
	compact,
	completeSummarization,
	generateSummary,
	generateSummaryWithUsage,
} from "../src/core/compaction/index.ts";

const { completeSimpleMock } = vi.hoisted(() => ({
	completeSimpleMock: vi.fn(),
}));

vi.mock("@earendil-works/pi-ai/compat", async (importOriginal) => {
	const actual = await importOriginal<typeof import("@earendil-works/pi-ai/compat")>();
	return {
		...actual,
		completeSimple: completeSimpleMock,
	};
});

function createModel(
	reasoning: boolean,
	maxTokens = 8192,
	compat?: Model<"anthropic-messages">["compat"],
): Model<"anthropic-messages"> {
	return {
		id: reasoning ? "reasoning-model" : "non-reasoning-model",
		name: reasoning ? "Reasoning Model" : "Non-reasoning Model",
		api: "anthropic-messages",
		provider: "anthropic",
		baseUrl: "https://api.anthropic.com",
		reasoning,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens,
		...(compat ? { compat } : {}),
	};
}

const mockSummaryResponse: AssistantMessage = {
	role: "assistant",
	content: [{ type: "text", text: "## Goal\nTest summary" }],
	api: "anthropic-messages",
	provider: "anthropic",
	model: "claude-sonnet-4-5",
	usage: {
		input: 10,
		output: 10,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 20,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	},
	stopReason: "stop",
	timestamp: Date.now(),
};

const mockToolCallResponse: AssistantMessage = {
	...mockSummaryResponse,
	content: [{ type: "toolCall", id: "tool-call-1", name: "read", arguments: { path: "README.md" } }],
	stopReason: "toolUse",
};

const messages: AgentMessage[] = [{ role: "user", content: "Summarize this.", timestamp: Date.now() }];

describe("generateSummary reasoning options", () => {
	beforeEach(() => {
		completeSimpleMock.mockReset();
		completeSimpleMock.mockResolvedValue(mockSummaryResponse);
	});

	it("uses the provided thinking level for reasoning-capable models", async () => {
		const result = await generateSummaryWithUsage(
			messages,
			createModel(true),
			2000,
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			"medium",
		);

		expect(result.text).toBe("## Goal\nTest summary");
		expect(result.usage).toEqual(mockSummaryResponse.usage);

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({
			reasoning: "medium",
			apiKey: "test-key",
		});
	});

	it("preserves the string result from generateSummary", async () => {
		await expect(generateSummary(messages, createModel(false), 2000, "test-key")).resolves.toBe(
			"## Goal\nTest summary",
		);
	});

	it("uses fresh routing sessions without prompt caching", async () => {
		await generateSummary(messages, createModel(false), 2000, "test-key");
		await generateSummary(messages, createModel(false), 2000, "test-key");

		const requestOptions = completeSimpleMock.mock.calls.map((call) => call[2]);
		expect(requestOptions).toHaveLength(2);
		expect(requestOptions.every((options) => options?.cacheRetention === "none")).toBe(true);

		const sessionIds = requestOptions.map((options) => options?.sessionId);
		expect(sessionIds[0]).not.toBe(sessionIds[1]);
	});

	it("honors caller-supplied routing session and tool choice without prompt caching", async () => {
		await completeSummarization(
			createModel(false),
			{ systemPrompt: "Summarize", messages: [] },
			{ sessionId: "current-routing-session", cacheRetention: "long", toolChoice: "auto" },
		);

		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({
			sessionId: "current-routing-session",
			cacheRetention: "none",
			toolChoice: "auto",
		});
	});

	it("preserves the standalone split-turn summary prompt", async () => {
		const preparation: CompactionPreparation = {
			firstKeptEntryId: "entry-keep",
			messagesToSummarize: [],
			turnPrefixMessages: messages,
			isSplitTurn: true,
			tokensBefore: 100,
			fileOps: { read: new Set(), written: new Set(), edited: new Set() },
			settings: { enabled: true, reserveTokens: 2000, keepRecentTokens: 20 },
		};

		await compact(preparation, createModel(false), "test-key");

		const requestContext = completeSimpleMock.mock.calls[0][1] as Context;
		const prompt = JSON.stringify(requestContext.messages);
		expect(prompt).toContain("This is the PREFIX of a turn that was too large to keep");
		expect(prompt).toContain("<conversation>");
	});

	it("rejects tool calls from conversation summaries", async () => {
		completeSimpleMock.mockResolvedValueOnce(mockToolCallResponse);

		await expect(generateSummaryWithUsage(messages, createModel(false), 2000, "test-key")).rejects.toThrow(
			"Summarization attempted to call a tool",
		);
	});

	it("rejects tool calls from split-turn summaries", async () => {
		completeSimpleMock.mockResolvedValueOnce(mockToolCallResponse);
		const preparation: CompactionPreparation = {
			firstKeptEntryId: "entry-keep",
			messagesToSummarize: [],
			turnPrefixMessages: messages,
			isSplitTurn: true,
			tokensBefore: 100,
			fileOps: { read: new Set(), written: new Set(), edited: new Set() },
			settings: { enabled: true, reserveTokens: 2000, keepRecentTokens: 20 },
		};

		await expect(compact(preparation, createModel(false), "test-key")).rejects.toThrow(
			"Turn prefix summarization attempted to call a tool",
		);
	});

	// A length stop used to be fatal, on the reasoning that partial text must not
	// become a session checkpoint. On a 32K window it is the ordinary case --
	// 2,242 of one arm's 3,018 compactions ended this way -- and each one threw
	// away a summary already paid for and left the context untouched. A fragment
	// this short is still refused, but it now buys a retry on half the prompt
	// rather than ending the compaction.
	it("retries a length-limited history summary instead of ending there", async () => {
		completeSimpleMock.mockResolvedValueOnce({
			...mockSummaryResponse,
			stopReason: "length",
			content: [{ type: "text", text: "partial" }],
		});

		const result = await generateSummaryWithUsage(messages, createModel(false), 2000, "test-key");

		expect(result.text).toBe("## Goal\nTest summary");
		expect(completeSimpleMock).toHaveBeenCalledTimes(2);
	});

	it("gives up on a length-limited history summary once the retries are spent", async () => {
		completeSimpleMock.mockResolvedValue({
			...mockSummaryResponse,
			stopReason: "length",
			content: [{ type: "text", text: "partial" }],
		});

		await expect(generateSummaryWithUsage(messages, createModel(false), 2000, "test-key")).rejects.toThrow(
			"generation hit the token cap",
		);
	});

	it("retries a length-limited split-turn summary too", async () => {
		completeSimpleMock.mockResolvedValueOnce({
			...mockSummaryResponse,
			stopReason: "length",
			content: [{ type: "text", text: "partial" }],
		});
		const preparation: CompactionPreparation = {
			firstKeptEntryId: "entry-keep",
			messagesToSummarize: [],
			turnPrefixMessages: messages,
			isSplitTurn: true,
			tokensBefore: 100,
			fileOps: { read: new Set(), written: new Set(), edited: new Set() },
			settings: { enabled: true, reserveTokens: 2000, keepRecentTokens: 20 },
		};

		const result = await compact(preparation, createModel(false), "test-key");

		expect(result.summary).toContain("Test summary");
	});

	it("does not set reasoning when thinking is off", async () => {
		await generateSummary(
			messages,
			createModel(true),
			2000,
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			"off",
		);

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({
			apiKey: "test-key",
		});
		expect(completeSimpleMock.mock.calls[0][2]).not.toHaveProperty("reasoning");
	});

	it("does not set reasoning for non-reasoning models", async () => {
		await generateSummary(
			messages,
			createModel(false),
			2000,
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			"medium",
		);

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({
			apiKey: "test-key",
		});
		expect(completeSimpleMock.mock.calls[0][2]).not.toHaveProperty("reasoning");
	});

	it("leaves Anthropic refusal fallback handling to pi-ai model metadata", async () => {
		await generateSummary(
			messages,
			createModel(true, 8192, {
				allowedFallbackModels: [
					{
						provider: "anthropic",
						model: "claude-opus-4-8",
						cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
					},
				],
			}),
			2000,
			"test-key",
		);

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		expect(completeSimpleMock.mock.calls[0][2]).not.toHaveProperty("refusalFallbacks");
	});

	it("does not set Anthropic refusal fallback for models without allowed fallback targets", async () => {
		await generateSummary(messages, createModel(true), 2000, "test-key");

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		expect(completeSimpleMock.mock.calls[0][2]).not.toHaveProperty("refusalFallbacks");
	});

	it("clamps compaction summary maxTokens to the model output cap", async () => {
		const preparation: CompactionPreparation = {
			firstKeptEntryId: "entry-keep",
			messagesToSummarize: messages,
			turnPrefixMessages: messages,
			isSplitTurn: true,
			tokensBefore: 600000,
			fileOps: { read: new Set(), written: new Set(), edited: new Set() },
			settings: { enabled: true, reserveTokens: 500000, keepRecentTokens: 20000 },
		};

		const result = await compact(preparation, createModel(false, 128000), "test-key");

		expect(result.usage).toEqual({
			...mockSummaryResponse.usage,
			input: 20,
			output: 20,
			totalTokens: 40,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		});
		// The model's own cap is 128,000 here and it is no longer what binds: a
		// summary is also capped at an eighth of the window it has to live inside,
		// because the next compaction summarises the summary. Left uncapped on a
		// 32K model that reached 13,107 tokens -- 40% of everything -- and grew
		// each round until it *was* the context: 63 turns, 49 compactions, a
		// context pinned at 32,554-32,659 against a 32,768 window, and turns
		// reporting `output: 1`. An eighth of 200,000 is 25,000.
		expect(completeSimpleMock.mock.calls.map((call) => call[2]?.maxTokens)).toEqual([25000, 25000]);
	});
});
