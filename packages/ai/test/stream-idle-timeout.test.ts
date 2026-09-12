import { describe, expect, it } from "vitest";
import { withIdleTimeout } from "../src/api/openai-completions.ts";

/**
 * The SDK's `timeout` covers getting a response, not keeping one. A request that
 * connects, returns headers and then stops sending chunks waits forever: across
 * one pair of 89-task benchmark arms, 25 trials ended in `AgentTimeoutError`
 * with their agent event stream stopped a median of 112 minutes before the
 * harness killed them, and eight of those had `message_start` as their final
 * event -- headers received, not one chunk after.
 */

async function* yields(values: number[], gapMs = 0): AsyncGenerator<number> {
	for (const value of values) {
		if (gapMs) await new Promise((resolve) => setTimeout(resolve, gapMs));
		yield value;
	}
}

/** Connects, hands over one chunk, then goes quiet for good. */
async function* stalls(): AsyncGenerator<number> {
	yield 1;
	await new Promise(() => {});
}

async function collect(source: AsyncIterable<number>): Promise<number[]> {
	const seen: number[] = [];
	for await (const value of source) seen.push(value);
	return seen;
}

describe("a stream that goes quiet", () => {
	it("fails instead of waiting on it forever", async () => {
		await expect(collect(withIdleTimeout(stalls(), 40))).rejects.toThrow(/idle for 0s with no data/);
	});

	it("hands over what arrived before the silence", async () => {
		const seen: number[] = [];
		await expect(
			(async () => {
				for await (const value of withIdleTimeout(stalls(), 40)) seen.push(value);
			})(),
		).rejects.toThrow();
		expect(seen).toEqual([1]);
	});

	it("leaves a stream that keeps talking alone", async () => {
		expect(await collect(withIdleTimeout(yields([1, 2, 3], 5), 200))).toEqual([1, 2, 3]);
	});

	it("measures the gap between chunks, not the whole stream", async () => {
		// Six chunks 30ms apart is 180ms of stream against a 100ms idle limit: the
		// limit is per gap, so this has to pass.
		expect(await collect(withIdleTimeout(yields([1, 2, 3, 4, 5, 6], 30), 100))).toHaveLength(6);
	});

	it("waits indefinitely when asked to", async () => {
		const race = await Promise.race([
			collect(withIdleTimeout(stalls(), 0)).then(() => "finished"),
			new Promise((resolve) => setTimeout(() => resolve("still waiting"), 60)),
		]);
		expect(race).toBe("still waiting");
	});
});

describe("the failure an idle stream produces", () => {
	it("is one the existing retry policy will take", async () => {
		// The watchdog is only half of it: pi already retries transient stream
		// failures at the session level (#3317, #6904), and this failure has to
		// look like one of those or the run ends on it instead of continuing.
		const { isRetryableAssistantError } = await import("../src/utils/retry.ts");
		let thrown = "";
		try {
			for await (const _ of withIdleTimeout(stalls(), 40)) {
				// consume until it gives up
			}
		} catch (error) {
			thrown = (error as Error).message;
		}
		expect(thrown).toContain("stream went idle");
		expect(
			isRetryableAssistantError({
				role: "assistant",
				content: [],
				stopReason: "error",
				errorMessage: thrown,
			} as never),
		).toBe(true);
	});

	it("is not confused with a rejected request", async () => {
		const { isRetryableAssistantError } = await import("../src/utils/retry.ts");
		expect(
			isRetryableAssistantError({
				role: "assistant",
				content: [],
				stopReason: "error",
				errorMessage: "insufficient_quota",
			} as never),
		).toBe(false);
	});
});
