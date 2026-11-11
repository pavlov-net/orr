"""Sizes of SPIR-V modules as the shader build emits them.

    python -I spv_size.py MODULE.spv [...]       instructions, ray queries, largest functions
    python -I spv_size.py --lines MODULE.spv     instructions by source file and most-copied lines
    python -I spv_size.py [--lines] --stage NAME the newest build of path/staged/NAME.spv

--lines needs a build with ORR_SHADER_DEBUG_INFO=lines. A line that shows many copies is a
body inlined at many call sites.
"""
import collections
import json
import os
import pathlib
import struct
import subprocess
import sys

OP_STRING, OP_NAME, OP_LINE, OP_FUNCTION, OP_FUNCTION_END, OP_NO_LINE = 7, 5, 8, 54, 56, 317
OP_RAY_QUERY_INITIALIZE, OP_RAY_QUERY_PROCEED, OP_LOOP_MERGE = 4473, 4477, 246


def instructions(path):
    data = open(path, "rb").read()
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    i = 5
    while i < len(words):
        count = words[i] >> 16
        if count == 0:
            break
        yield words[i] & 0xFFFF, words[i + 1 : i + count]
        i += count


def text(operands):
    return struct.pack("<%dI" % len(operands), *operands).split(b"\0")[0].decode("utf8", "replace")


def summary(path):
    ops = collections.Counter()
    names, functions, current = {}, [], None
    for op, operands in instructions(path):
        ops[op] += 1
        if op == OP_NAME:
            names[operands[0]] = text(operands[1:])
        elif op == OP_FUNCTION:
            current = [operands[1], 0]
        elif op == OP_FUNCTION_END and current:
            functions.append(current)
            current = None
        if current:
            current[1] += 1
    print("%s: %d instructions, %d functions, %d ray query starts, %d proceed sites, %d loops" % (
        os.path.basename(path), sum(ops.values()), len(functions),
        ops[OP_RAY_QUERY_INITIALIZE], ops[OP_RAY_QUERY_PROCEED], ops[OP_LOOP_MERGE]))
    for function, count in sorted(functions, key=lambda f: -f[1])[:8]:
        print("  %8d  %s" % (count, names.get(function, "%%%d" % function)[:90]))


def lines(path):
    strings, by_file, by_line, copies = {}, collections.Counter(), collections.Counter(), collections.Counter()
    current, inside, total = None, False, 0
    for op, operands in instructions(path):
        if op == OP_STRING:
            strings[operands[0]] = text(operands[1:])
        elif op == OP_FUNCTION:
            inside, current = True, None
        elif op == OP_FUNCTION_END:
            inside = False
        elif op == OP_LINE:
            key = (operands[0], operands[1])
            if key != current:
                copies[key] += 1
            current = key
        elif op == OP_NO_LINE:
            current = None
        elif inside:
            total += 1
            if current:
                by_file[current[0]] += 1
                by_line[current] += 1

    def name(file):
        return strings.get(file, "?").replace("\\", "/").split("/crates/")[-1][-56:]

    if not by_line:
        sys.exit("no line info: build with ORR_SHADER_DEBUG_INFO=lines")
    print("%s: %d instructions in functions" % (os.path.basename(path), total))
    print("by source file:")
    for file, count in by_file.most_common(20):
        print("  %8d %5.1f%%  %s" % (count, 100 * count / total, name(file)))
    print("lines with the most instructions over all their copies:")
    for key, count in by_line.most_common(25):
        print("  %8d  %4d copies  %s:%d" % (count, copies[key], name(key[0]), key[1]))


def newest_stage(stage):
    # The script lives in <checkout>/.claude/skills/<skill>/scripts, so this finds the build of
    # the checkout the skill came from, worktrees included.
    checkout = pathlib.Path(__file__).resolve().parents[4]
    metadata = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=checkout, capture_output=True, text=True, check=True)
    build = pathlib.Path(json.loads(metadata.stdout)["build_directory"])
    found = list(build.glob("*/build/orr_lighting/*/out/path/staged/%s.spv" % stage))
    if not found:
        sys.exit("no path/staged/%s.spv under %s: build the lighting crate first" % (stage, build))
    module = max(found, key=lambda path: path.stat().st_mtime)
    print(module)
    return str(module)


if __name__ == "__main__":
    args = sys.argv[1:]
    by_lines = args[:1] == ["--lines"]
    if by_lines:
        args = args[1:]
    if args[:1] == ["--stage"] and len(args) == 2:
        args = [newest_stage(args[1])]
    if not args or args[0].startswith("--") or (by_lines and len(args) != 1):
        sys.exit(__doc__)
    if by_lines:
        lines(args[0])
    else:
        for module in args:
            summary(module)
