import { randomBytes } from "node:crypto";
import { createWriteStream, type WriteStream } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, type TruncationResult, truncateTail } from "./truncate.ts";

export interface OutputAccumulatorOptions {
	maxLines?: number;
	maxBytes?: number;
	tempFilePrefix?: string;
	/**
	 * Share of the line and byte budgets spent on the output's first lines when
	 * it is truncated, for snapshots that ask for them. 0 keeps only the tail.
	 */
	headShare?: number;
}

export interface OutputSnapshot {
	content: string;
	truncation: TruncationResult;
	fullOutputPath?: string;
}

function defaultTempFilePath(prefix: string): string {
	const id = randomBytes(8).toString("hex");
	return join(tmpdir(), `${prefix}-${id}.log`);
}

function byteLength(text: string): number {
	return Buffer.byteLength(text, "utf-8");
}

/**
 * Incrementally tracks streaming output with bounded memory.
 *
 * Appends decode chunks with a streaming UTF-8 decoder, keeps only a decoded
 * tail for display snapshots, and opens a temp file when the full output needs
 * to be preserved.
 */
export class OutputAccumulator {
	private readonly maxLines: number;
	private readonly maxBytes: number;
	private readonly maxRollingBytes: number;
	private readonly tempFilePrefix: string;
	private readonly decoder = new TextDecoder();

	private rawChunks: Buffer[] = [];
	private tailText = "";
	private tailBytes = 0;
	private tailStartsAtLineBoundary = true;
	private totalRawBytes = 0;
	private totalDecodedBytes = 0;
	private completedLines = 0;
	private totalLines = 0;
	private currentLineBytes = 0;
	private hasOpenLine = false;
	private finished = false;

	private tempFilePath: string | undefined;
	private tempFileStream: WriteStream | undefined;

	private readonly headMaxLines: number;
	private readonly headMaxBytes: number;
	private headText = "";
	private headBytes = 0;
	private headLines = 0;
	private headPending = "";
	private headClosed: boolean;

	constructor(options: OutputAccumulatorOptions = {}) {
		this.maxLines = options.maxLines ?? DEFAULT_MAX_LINES;
		this.maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;
		this.maxRollingBytes = Math.max(this.maxBytes * 2, 1);
		this.tempFilePrefix = options.tempFilePrefix ?? "pi-output";
		const headShare = Math.min(Math.max(options.headShare ?? 0, 0), 0.5);
		this.headMaxLines = Math.floor(this.maxLines * headShare);
		this.headMaxBytes = Math.floor(this.maxBytes * headShare);
		this.headClosed = this.headMaxLines === 0 || this.headMaxBytes === 0;
	}

	append(data: Buffer): void {
		if (this.finished) {
			throw new Error("Cannot append to a finished output accumulator");
		}

		this.totalRawBytes += data.length;
		this.appendDecodedText(this.decoder.decode(data, { stream: true }));

		if (this.tempFileStream || this.shouldUseTempFile()) {
			this.ensureTempFile();
			this.tempFileStream?.write(data);
		} else if (data.length > 0) {
			this.rawChunks.push(data);
		}
	}

	finish(): void {
		if (this.finished) {
			return;
		}
		this.finished = true;
		this.appendDecodedText(this.decoder.decode());
		if (this.shouldUseTempFile()) {
			this.ensureTempFile();
		}
	}

	/**
	 * `withHead` keeps the output's first lines ahead of the tail when it is
	 * truncated, out of the same budget. It is for the result the model reads;
	 * a live view wants only the latest lines.
	 */
	snapshot(options: { persistIfTruncated?: boolean; withHead?: boolean } = {}): OutputSnapshot {
		const truncated = this.totalLines > this.maxLines || this.totalDecodedBytes > this.maxBytes;
		const truncation = (truncated && options.withHead && this.headAndTail()) || this.tailOnly(truncated);

		if (options.persistIfTruncated && truncation.truncated) {
			this.ensureTempFile();
		}

		return {
			content: truncation.content,
			truncation,
			fullOutputPath: this.tempFilePath,
		};
	}

	private tailOnly(
		truncated: boolean,
		budget = { maxLines: this.maxLines, maxBytes: this.maxBytes },
	): TruncationResult {
		const tailTruncation = truncateTail(this.getSnapshotText(), budget);
		const truncatedBy = truncated
			? (tailTruncation.truncatedBy ?? (this.totalDecodedBytes > this.maxBytes ? "bytes" : "lines"))
			: null;
		return {
			...tailTruncation,
			truncated,
			truncatedBy,
			totalLines: this.totalLines,
			totalBytes: this.totalDecodedBytes,
			maxLines: this.maxLines,
			maxBytes: this.maxBytes,
		};
	}

	/**
	 * The output's first lines, a marker for what was left out, then the tail --
	 * or undefined when that would not show anything the tail alone does not.
	 */
	private headAndTail(): TruncationResult | undefined {
		if (this.headLines === 0) return undefined;
		const tail = this.tailOnly(true, {
			maxLines: this.maxLines - this.headLines,
			maxBytes: this.maxBytes - this.headBytes,
		});
		// A single line too long for the budget has no first lines to keep.
		if (tail.lastLinePartial) return undefined;
		const tailStart = this.totalLines - tail.outputLines + 1;
		const omitted = tailStart - this.headLines - 1;
		if (omitted <= 0) return undefined;
		return {
			...tail,
			content: `${this.headText}\n[... ${omitted} ${omitted === 1 ? "line" : "lines"} omitted ...]\n\n${tail.content}`,
			outputLines: this.headLines + tail.outputLines,
			outputBytes: this.headBytes + tail.outputBytes,
			headLines: this.headLines,
		};
	}

	async closeTempFile(): Promise<void> {
		if (!this.tempFileStream) {
			return;
		}

		const stream = this.tempFileStream;
		this.tempFileStream = undefined;

		await new Promise<void>((resolve, reject) => {
			const onError = (error: Error) => {
				stream.off("finish", onFinish);
				reject(error);
			};
			const onFinish = () => {
				stream.off("error", onError);
				resolve();
			};
			stream.once("error", onError);
			stream.once("finish", onFinish);
			stream.end();
		});
	}

	getLastLineBytes(): number {
		return this.currentLineBytes;
	}

	private appendDecodedText(text: string): void {
		if (text.length === 0) {
			return;
		}

		if (!this.headClosed) {
			this.captureHead(text);
		}

		const bytes = byteLength(text);
		this.totalDecodedBytes += bytes;
		this.tailText += text;
		this.tailBytes += bytes;
		if (this.tailBytes > this.maxRollingBytes * 2) {
			this.trimTail();
		}

		let newlines = 0;
		let lastNewline = -1;
		for (let i = text.indexOf("\n"); i !== -1; i = text.indexOf("\n", i + 1)) {
			newlines++;
			lastNewline = i;
		}
		if (newlines === 0) {
			this.currentLineBytes += bytes;
			this.hasOpenLine = true;
		} else {
			this.completedLines += newlines;
			const tail = text.slice(lastNewline + 1);
			this.currentLineBytes = byteLength(tail);
			this.hasOpenLine = tail.length > 0;
		}
		this.totalLines = this.completedLines + (this.hasOpenLine ? 1 : 0);
	}

	/** Keeps whole lines from the start of the output until the head budget is spent. */
	private captureHead(text: string): void {
		this.headPending += text;
		let newline = this.headPending.indexOf("\n");
		while (newline !== -1) {
			const line = this.headPending.slice(0, newline + 1);
			const bytes = byteLength(line);
			if (this.headLines + 1 > this.headMaxLines || this.headBytes + bytes > this.headMaxBytes) {
				this.closeHead();
				return;
			}
			this.headText += line;
			this.headBytes += bytes;
			this.headLines++;
			this.headPending = this.headPending.slice(newline + 1);
			newline = this.headPending.indexOf("\n");
		}
		if (byteLength(this.headPending) > this.headMaxBytes - this.headBytes) {
			this.closeHead();
		}
	}

	private closeHead(): void {
		this.headClosed = true;
		this.headPending = "";
	}

	private trimTail(): void {
		const buffer = Buffer.from(this.tailText, "utf-8");
		if (buffer.length <= this.maxRollingBytes) {
			this.tailBytes = buffer.length;
			return;
		}

		let start = buffer.length - this.maxRollingBytes;
		while (start < buffer.length && (buffer[start] & 0xc0) === 0x80) {
			start++;
		}

		this.tailStartsAtLineBoundary = start === 0 ? this.tailStartsAtLineBoundary : buffer[start - 1] === 0x0a;
		this.tailText = buffer.subarray(start).toString("utf-8");
		this.tailBytes = byteLength(this.tailText);
	}

	private getSnapshotText(): string {
		if (this.tailStartsAtLineBoundary) {
			return this.tailText;
		}

		const firstNewline = this.tailText.indexOf("\n");
		return firstNewline === -1 ? this.tailText : this.tailText.slice(firstNewline + 1);
	}

	private shouldUseTempFile(): boolean {
		return (
			this.totalRawBytes > this.maxBytes || this.totalDecodedBytes > this.maxBytes || this.totalLines > this.maxLines
		);
	}

	private ensureTempFile(): void {
		if (this.tempFilePath) {
			return;
		}
		this.tempFilePath = defaultTempFilePath(this.tempFilePrefix);
		this.tempFileStream = createWriteStream(this.tempFilePath);
		for (const chunk of this.rawChunks) {
			this.tempFileStream.write(chunk);
		}
		this.rawChunks = [];
	}
}
