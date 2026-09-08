import { afterEach, describe, expect, it } from "vitest";
import { deadlineFromEnv, maxOutputTokensFromEnv } from "../../src/core/sdk.ts";

// Nothing inside pi knows how long it is allowed to run. A batch harness does:
// a benchmark trial is given a budget and killed at it, so the run ends
// mid-tool-call with whatever was on disk at that instant. Six of pi's
// twenty-four failed Terminal-Bench 2.1 trials ended that way, while the median
// successful one used a fraction of the budget it had. An environment variable
// is how a harness launching `pi --print` hands the number over without pi
// having to know what a harness is.

const KEY = "PI_TIME_BUDGET_SEC";
const original = process.env[KEY];

afterEach(() => {
	if (original === undefined) delete process.env[KEY];
	else process.env[KEY] = original;
});

function withBudget(value: string | undefined): number | undefined {
	if (value === undefined) delete process.env[KEY];
	else process.env[KEY] = value;
	return deadlineFromEnv();
}

describe("PI_TIME_BUDGET_SEC", () => {
	it("turns a budget into a deadline that far ahead", () => {
		const before = Date.now();
		const deadline = withBudget("7200");
		expect(deadline).toBeDefined();
		expect(deadline! - before).toBeGreaterThanOrEqual(7_200_000 - 50);
		expect(deadline! - before).toBeLessThan(7_200_000 + 5_000);
	});

	it("is unset for an interactive session, where the person is the deadline", () => {
		expect(withBudget(undefined)).toBeUndefined();
	});

	it("ignores a value that cannot be a budget rather than inventing one", () => {
		// A deadline in the past would end the run before it began, which is a
		// worse failure than having no deadline at all.
		for (const bad of ["", "0", "-1", "soon", "NaN"]) {
			expect(withBudget(bad)).toBeUndefined();
		}
	});
});

describe("PI_MAX_OUTPUT_TOKENS", () => {
	// A provider reserves inference capacity by `max_tokens`, so asking for a
	// large one costs queueing whether or not the tokens are used. Claude Code
	// caps its default at 8K against a p99 output of 4,911 tokens, accepts under
	// 1% truncation, and gives those a clean retry at the model's ceiling --
	// the retry the agent loop now performs.
	//
	// The number is deployment-specific, which is why this is a setting rather
	// than a new default: over 10,225 assistant turns on one self-hosted 27B
	// deployment, p50 output is 461 tokens, p95 is 5,699 and p99 is 16,161 --
	// three times Claude Code's, so their 8K truncates 3% here where 16K
	// truncates 0.4%.
	const MKEY = "PI_MAX_OUTPUT_TOKENS";
	const mOriginal = process.env[MKEY];

	afterEach(() => {
		if (mOriginal === undefined) delete process.env[MKEY];
		else process.env[MKEY] = mOriginal;
	});

	function withCap(value: string | undefined) {
		if (value === undefined) delete process.env[MKEY];
		else process.env[MKEY] = value;
		return maxOutputTokensFromEnv();
	}

	it("is unset by default, leaving pi's behaviour alone", () => {
		expect(withCap(undefined)).toBeUndefined();
	});

	it("carries the ceiling when one is asked for", () => {
		expect(withCap("16384")).toBe(16384);
	});

	it("ignores a value that cannot be a ceiling", () => {
		// A zero or negative cap would truncate every response at once.
		for (const bad of ["", "0", "-1", "lots", "NaN"]) {
			expect(withCap(bad)).toBeUndefined();
		}
	});
});
