import { describe, expect, it } from "vitest";
import { DEFAULT_BASH_TIMEOUT_SECONDS, resolveTimeoutMs } from "../src/core/tools/bash.ts";

/**
 * A command with no timeout takes the trial with it. Across one pair of 89-task
 * arms, 25 trials ended in `AgentTimeoutError` with their agent event stream
 * stopped a median of 112 minutes before harbor killed them, and eight of those
 * were a bash command that never returned -- `grep -rl ... /` across the whole
 * filesystem, `vncsnapshot` against a VM that never booted.
 * `install-windows-3.11` ran 48 seconds of turns and then held its container for
 * eight hours.
 */

describe("the timeout a command gets when it does not ask", () => {
	it("is ten minutes rather than forever", () => {
		expect(resolveTimeoutMs(undefined)).toBe(600_000);
		expect(DEFAULT_BASH_TIMEOUT_SECONDS).toBe(600);
	});

	it("is long enough for the builds this benchmark runs", () => {
		// Claude Code's two minutes would kill a five-minute compile, which takes
		// away trials that pass today; ten still turns an eight-hour hang into one
		// lost turn.
		expect(DEFAULT_BASH_TIMEOUT_SECONDS).toBeGreaterThanOrEqual(300);
	});

	it("still lets a command ask for more or less", () => {
		expect(resolveTimeoutMs(5)).toBe(5_000);
		expect(resolveTimeoutMs(3600)).toBe(3_600_000);
	});

	it("still refuses a nonsense timeout rather than falling back to the default", () => {
		expect(() => resolveTimeoutMs(0)).toThrow(/finite number of seconds/);
		expect(() => resolveTimeoutMs(-1)).toThrow(/finite number of seconds/);
		expect(() => resolveTimeoutMs(Number.NaN)).toThrow(/finite number of seconds/);
		expect(() => resolveTimeoutMs(3_000_000)).toThrow(/maximum is/);
	});
});
