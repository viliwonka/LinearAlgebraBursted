# DEVLOG — Hash
Code comments state contracts only; history lives here (see CLAUDE.md).

## Hash.Shared.cs
- 2026-07-13 | Codegen hazard (moved from a "NOTE FOR EDITORS" comment): this file must never
  contain the literal proxy-token substrings that name the sibling per-type files (the
  float/double token or the int/short/long token) -- either one appearing anywhere in this
  file's text, even in a comment, flips TemplateConverter's singular-file detection and this
  file would silently get copy-mangled once per int-family type. Refer to the sibling files by
  description ("the float/double file", "the int-family file") instead of by name. (was
  Hash.Shared.cs:10-15)

## Hash.fProxy.cs / Hash.iProxy.cs
- 2026-07-13 | Codegen mechanics (moved from inline comments): the row/col hash dest is always a
  uint buffer regardless of A's element type. `uintN`/`uintMxN` are codegen OUTPUTS (in
  Hash.fProxy.cs, the iProxy token is always chosen as uint for both float and double; in
  Hash.iProxy.cs, the dest is spelled via the real `iProxyN` placeholder token but immediately
  CHOSEN to the fixed literal "uintN" for every one of the file's 4 generated slots --
  int/short/long/uint), not a type that exists in TemplateSource's own standalone compile, so it
  is emitted via a choose marker rather than hand-written. This keeps int-sourced/short-sourced/
  long-sourced rowHashes/colHashes correctly returning a uint buffer instead of accidentally
  tracking A's own element type. (was Hash.fProxy.cs:40-43, Hash.iProxy.cs:24-30)
