import { afterEach, describe, expect, it } from "vitest";
import { deadlineFromEnv } from "../../src/core/sdk.ts";

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
