## Introduction

[![Report card](http://goreportcard.com/badge/danieldk/golinear)](http://goreportcard.com/report/danieldk/golinear)
[![GoDoc](https://godoc.org/gopkg.in/danieldk/golinear.v1?status.svg)](https://godoc.org/gopkg.in/danieldk/golinear.v1)

golinear is a package for training and using linear classifiers in the Go
programming language (golang).

## Installation

To use this package, you need the
[liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) library. On Mac
OS X, you can install this library with
[Homebrew](http://mxcl.github.com/homebrew/):

    brew install liblinear

Ubuntu and Debian provide packages for *liblinear*. However, at the time of
writing (July 2, 2014), these were serverly outdated. This package requires
version 1.9 or later.

This latest API-stable version (v1) can be installed with the <tt>go</tt>
command:

    go get gopkg.in/danieldk/golinear.v1

or included in your source code:

    import "gopkg.in/danieldk/golinear.v1"

The package documentation is available at: http://godoc.org/gopkg.in/danieldk/golinear.v1

### OpenMP

If you wish to use *liblinear* with OpenMP support for multicore processing, 
please use this command to install the package:

    CGO_LDFLAGS="-lgomp" CGO_CFLAGS="-DCV_OMP" go get github.com/danieldk/golinear

## Plans

1. Port classification to Go.
2. Port training to Go.

We will take a pragmatic approach to porting code to Go: if the performance penalty is minor,
ported code will flow to the main branch. Otherwise, we will keep it around until the performance
is good enough.

## Examples

Examples for using golinear can be found at:

https://github.com/danieldk/golinear-examples
