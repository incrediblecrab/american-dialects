// Assert package-lock.json can be replayed on the machine that builds the site.
//
// The lockfile is written on macOS and `npm ci` replays it on Linux, and those
// two runs do not want the same files. Packages that ship a compiled binary
// publish one tarball per platform and depend on all of them optionally, so
// that npm installs only the one it needs: sharp has @img/sharp-darwin-arm64
// and @img/sharp-linux-x64, esbuild has @esbuild/darwin-arm64 and
// @esbuild/linux-x64. Whether every one of those lands in the lockfile depends
// on how the tree was resolved, and a lockfile that records only the macOS
// binary installs fine here and then fails in CI, which is a failure that
// cannot be reproduced on the machine that caused it.
//
// So this checks statically what would otherwise need a Linux machine to
// discover: every dependency that ships one tarball per platform must include
// the tarball a linux/x64 runner would ask for.
//
// Two kinds of optional dependency are deliberately not families and are
// skipped. A package that is itself gated off Linux, such as
// @img/sharp-darwin-arm64, is right to depend only on darwin things, because
// nothing on the runner will ever reach it. And a lone platform-gated
// dependency is a capability rather than a family: vite depends optionally on
// fsevents, which is a macOS file watcher with no Linux counterpart, so
// demanding a Linux member would be demanding a package that does not exist.
// A family is therefore a parent that installs on the runner and offers the
// same thing more than one way.
//
// Run it before pushing. `npm ci` proves the lockfile agrees with
// package.json; this proves it agrees with the runner.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const TARGET = { os: "linux", cpu: "x64" };

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const lock = JSON.parse(readFileSync(join(root, "package-lock.json"), "utf8"));
const entries = lock.packages ?? {};

// npm writes these as arrays that are either an allow list ("darwin") or a
// deny list ("!win32"), and omits the field when the package runs anywhere.
function accepts(field, want) {
  if (!field) return true;
  const values = Array.isArray(field) ? field : [field];
  const denied = values.filter((v) => v.startsWith("!"));
  if (denied.length > 0) return !denied.some((v) => v.slice(1) === want);
  return values.includes(want);
}

function runsOnTarget(entry) {
  return accepts(entry.os, TARGET.os) && accepts(entry.cpu, TARGET.cpu);
}

function isPlatformGated(entry) {
  return Boolean(entry?.os || entry?.cpu);
}

// A dependency can sit at the hoisted root or nested under whichever package
// pulled it in, so resolve by name rather than by a path we would have to guess.
function findByName(name) {
  const suffix = `node_modules/${name}`;
  return Object.entries(entries)
    .filter(([path]) => path === suffix || path.endsWith(`/${suffix}`))
    .map(([path, entry]) => ({ path, ...entry }));
}

const failures = [];
const checked = [];

for (const [parentPath, parent] of Object.entries(entries)) {
  const optional = parent.optionalDependencies;
  if (!optional) continue;

  // A parent the runner never installs cannot strand the runner.
  if (!runsOnTarget(parent)) continue;

  const gated = [];
  for (const name of Object.keys(optional)) {
    for (const found of findByName(name)) {
      if (isPlatformGated(found)) gated.push({ name, ...found });
    }
  }

  // One gated dependency is a capability, not a per-platform family.
  if (gated.length < 2) continue;

  const owner = parentPath === "" ? "(root)" : parentPath;
  const usable = gated.filter(runsOnTarget);
  checked.push({ owner, total: gated.length, usable: usable.length });

  if (usable.length === 0) {
    failures.push(
      `${owner} ships ${gated.length} per-platform builds and none of them ` +
        `install on ${TARGET.os}/${TARGET.cpu}.\n` +
        `    present: ${gated.map((g) => g.name).join(", ")}\n` +
        `    fix: rm -rf node_modules package-lock.json && npm install`,
    );
  }
}

for (const { owner, total, usable } of checked) {
  console.log(`  ${String(usable).padStart(2)}/${String(total).padEnd(2)} on ${TARGET.os}/${TARGET.cpu}  ${owner}`);
}

if (failures.length > 0) {
  console.error(`\nlockfile will not replay on ${TARGET.os}/${TARGET.cpu}:\n`);
  for (const f of failures) console.error(`  ${f}\n`);
  process.exit(1);
}

console.log(
  `\n${checked.length} per-platform ${
    checked.length === 1 ? "family" : "families"
  } all resolve on ${TARGET.os}/${TARGET.cpu}.`,
);
