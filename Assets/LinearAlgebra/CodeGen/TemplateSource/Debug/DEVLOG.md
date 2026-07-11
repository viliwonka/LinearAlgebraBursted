# DEVLOG — Debug
Code comments state contracts only; history lives here (see CLAUDE.md).

## Export.bool.cs / Export.fProxy.cs / Export.iProxy.cs
- 2026-07-11 | Codegen hazard shared by the Export.*.cs family: these are singularFile-style
  exporters copied through the code generator unchanged (not per-type multiplied). Export.bool.cs
  must never contain either of the code generator's two per-type placeholder spellings (see
  GenUtils.cs) -- doing so would make TemplateConverter.Execute treat it as a multiplying file
  instead of copying it through unchanged, and since this filename doesn't contain either
  placeholder, the copies would collide on the same output path. Export.iProxy.cs's per-type
  "choose" codegen marker (see GenUtils.cs / proxyStructs.cs) casts to (int)/(short)/(long)/(uint)
  at the proxy-compile stage and an identity cast after codegen substitution; the literal
  choose-marker token must never be written out inside a comment in that file, since the codegen
  parser is content-sensitive and would try to expand it. (was Export.bool.cs:6-21,
  Export.iProxy.cs:7-23)
