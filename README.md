# Linear Algebra Library for Unity

## Description
A linear algebra library for Unity, fully written in Burst. It's designed to be a natural extension of Unity.Mathematics, offering a bit more of functionalities. Currently in an experimental stage and not yet ready for production use.

## Installation

To open the repo in Unity, follow these steps:

1. Clone the repo
2. Open the project in Unity

To use library in your own project:

1. Clone the repo to separate project
2. Copy Assets/LinearAlgebra/Source into your own project

## Usage

Here's a simple example to get started with the library:

```csharp
using LinearAlgebra;

// memory management struct
var arena = new Arena(Allocator.Persistent);

int dim = 128;
// creates a zero vector of 128 dimensions 
floatN vecA = arena.floatVec(dim);
// creates a vector of 128 dimensions with all elements set to 1
floatN vecB = arena.floatVec(dim, 1f);

// add per component (will allocate a new temporary vec)
floatN vecAdd = vecA + vecB;
// mul per component (will allocate a new temporary vec)
floatN vecMul = vecA * vecB;

// create identity matrix
floatMxN matI = arena.floatIdentityMatrix(dim);
floatMxN matRand = arena.floatRandomMatrix(16, 16);

// per component sum, allocates new matrix
floatMxN compSumMat = matI + matRand;

// adds 1f to compSumMat inplace, allocating nothing
floatOP.addInpl(compSumMat, 1f);

// mulls matI into compSumMat inplace, allocating nothing 
floatOP.mulInpl(compSumMat, matI)

// creates random matrix with range from -3f to 3f
floatMxN A = arena.floatRandomDiagonalMatrix(dim, dim, -3f, 3f);
floatMxN B = arena.floatRandomDiagonalMatrix(dim, dim, -3f, 3f);

// dot multiply A and B, will allocate new matrix
C = floatOP.dot(A, B);

// dispose all allocated vectors/matrices
arena.Dispose();
```

