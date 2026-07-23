# Comp - element-wise operations

Per component operations for vectors/matrices of next types:

- `float` / `double`,
- `int` / `uint`,
- `bool`,
- `short`,
- `long`

All mutating methods have suffix `InPlace`.

Not all methods/functions are written here. Many of the mentioned are inplace with `InPlace` suffix, without writing it.

## float/double


- **Arithmetic**: 
  - `add` / `sub`,
  - `mul` / `div` / `mod`,
  - `addScaled`,
  - `clamp`,
  - `zero` / `fill(s)` (also on int/uint/long/short)

- **Functions:** 
  - `abs`, 
  - `sign`, 
  - `sqrt` / `rsqrt`,
  - `sin` / `cos` / `tan` / `asin` / `acos` / `atan`, 
  - `exp` , `exp2`
  - `exp10` , `log`, `log2` / `log10`,
  - `ceil` / `floor`/
  - `round`, `relu`, `pow(int exponent)`, `rcp`, `frac`,
  - `saturate`, `degrees`/`radians`.

- **Two-buffer/interpolation:** 
  - `lerp(a,b,t)`, 
  - `unlerp`, 
  - `smoothstep`,
  - `step(edge)`, 
  - `mad(a,b,c)`, 
  - `remap(oldMin,oldMax,newMin,newMax)`,
  - `atan2(y,x)`, `min`/`max`, 

All of the above are in-place kernels over buffers you own — allocate first (see
[dense-types](dense-types.md)), then mutate. To keep an operand, copy it
(`new floatN(in a, Allocator.Temp)`) and mutate the copy.

## bool

Logical ops
- bitwise ops `not`, `and`, `or`, `xor`
- reduce ops `any`, `all`

Compare ops
- `equals`, `notEquals`

## Integer bit ops

`int` / `short` / `long` / `uint` add bitwise ops on top of the arithmetic set:

- `bitwiseAnd` / `bitwiseOr`,
- `bitwiseXor` / `bitwiseComplement`,
- `bitwiseLeftShift` / `bitwiseRightShift`, 
- `ror` / `rol`, 
- `countbits`,
- `tzcnt` / `lzcnt`,
- `reversebits`, 
- `ceilpow2`
