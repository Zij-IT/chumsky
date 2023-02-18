//! A zero-copy implementation of [`Parser`](super::Parser)
//!
//! This will likely be moving to the crate root at some point and entirely replacing the current
//! parser implementation.

macro_rules! go_extra {
    ( $O :ty ) => {
        #[inline(always)]
        fn go_emit(&self, inp: &mut InputRef<'a, '_, I, E>) -> PResult<Emit, $O, E::Error> {
            Parser::<I, $O, E>::go::<Emit>(self, inp)
        }
        #[inline(always)]
        fn go_check(&self, inp: &mut InputRef<'a, '_, I, E>) -> PResult<Check, $O, E::Error> {
            Parser::<I, $O, E>::go::<Check>(self, inp)
        }
    };
}

macro_rules! go_cfg_extra {
    ( $O :ty ) => {
        #[inline(always)]
        fn go_emit_cfg(
            &self,
            inp: &mut InputRef<'a, '_, I, E>,
            cfg: Self::Config,
        ) -> PResult<Emit, $O, E::Error> {
            ConfigParser::<I, $O, E>::go_cfg::<Emit>(self, inp, cfg)
        }
        #[inline(always)]
        fn go_check_cfg(
            &self,
            inp: &mut InputRef<'a, '_, I, E>,
            cfg: Self::Config,
        ) -> PResult<Check, $O, E::Error> {
            ConfigParser::<I, $O, E>::go_cfg::<Check>(self, inp, cfg)
        }
    };
}

mod blanket;
pub mod combinator;
pub mod container;
pub mod error;
pub mod extra;
pub mod input;
pub mod pratt;
pub mod primitive;
pub mod recovery;
pub mod recursive;
#[cfg(feature = "regex")]
pub mod regex;
pub mod span;
pub mod text;

/// Commonly used functions, traits and types.
///
/// *Listen, three eyes,” he said, “don’t you try to outweird me, I get stranger things than you free with my breakfast
/// cereal.”*
pub mod prelude {
    pub use super::{
        error::{EmptyErr, Error as _, Rich, Simple},
        extra,
        pratt,
        primitive::{
            any, choice, empty, end, group, just, map_ctx, none_of, one_of, take_until, todo,
        },
        recovery::{nested_delimiters, skip_until},
        recursive::{recursive, Recursive},
        // select,
        span::{SimpleSpan, Span as _},
        text,
        Boxed,
        ConfigIterParser,
        ConfigParser,
        IterParser,
        ParseResult,
        Parser,
    };
}

use alloc::{
    boxed::Box,
    rc::{Rc, Weak},
    string::String,
    vec,
    vec::Vec,
};
use core::{
    borrow::Borrow,
    cmp::{Eq, Ordering},
    fmt,
    hash::Hash,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Range, RangeFrom},
    str::FromStr,
};
use hashbrown::HashMap;

use self::{
    combinator::*,
    container::*,
    error::Error,
    extra::ParserExtra,
    input::{Input, InputRef, SliceInput, StrInput},
    pratt::Pratt,
    prelude::*,
    recovery::RecoverWith,
    span::Span,
    text::*,
};

// TODO: Remove this when MaybeUninit transforms to/from arrays stabilize in any form
trait MaybeUninitExt<T>: Sized {
    /// Identical to the unstable [`MaybeUninit::uninit_array`]
    fn uninit_array<const N: usize>() -> [Self; N];

    /// Identical to the unstable [`MaybeUninit::array_assume_init`]
    unsafe fn array_assume_init<const N: usize>(uninit: [Self; N]) -> [T; N];
}

impl<T> MaybeUninitExt<T> for MaybeUninit<T> {
    fn uninit_array<const N: usize>() -> [Self; N] {
        // SAFETY: Output type is entirely uninhabited - IE, it's made up entirely of `MaybeUninit`
        unsafe { MaybeUninit::uninit().assume_init() }
    }

    unsafe fn array_assume_init<const N: usize>(uninit: [Self; N]) -> [T; N] {
        let out = (&uninit as *const [Self; N] as *const [T; N]).read();
        core::mem::forget(uninit);
        out
    }
}

/// The result of calling [`Parser::go`]
pub type PResult<M, O, E> = Result<<M as Mode>::Output<O>, Located<E>>;

/// The result of running a [`Parser`]. Can be converted into a [`Result`] via
/// [`ParseResult::into_result`] for when you only care about success or failure, or into distinct
/// error and output via [`ParseResult::into_output_errors`]
#[derive(Debug, Clone, PartialEq)]
pub struct ParseResult<T, E> {
    output: Option<T>,
    errs: Vec<E>,
}

impl<T, E> ParseResult<T, E> {
    pub(crate) fn new(output: Option<T>, errs: Vec<E>) -> ParseResult<T, E> {
        ParseResult { output, errs }
    }

    /// Whether this result contains output
    pub fn has_output(&self) -> bool {
        self.output.is_some()
    }

    /// Whether this result has any errors
    pub fn has_errors(&self) -> bool {
        !self.errs.is_empty()
    }

    /// Get a reference to the output of this result, if it exists
    pub fn output(&self) -> Option<&T> {
        self.output.as_ref()
    }

    /// Get a slice containing the parse errors for this result. The slice will be empty
    /// if there are no errors.
    pub fn errors(&self) -> impl ExactSizeIterator<Item = &E> {
        self.errs.iter()
    }

    /// Convert this `ParseResult` into an option containing the output, if any exists
    pub fn into_output(self) -> Option<T> {
        self.output
    }

    /// Convert this `ParseResult` into a vector containing any errors. The vector will be empty
    /// if there were no errors.
    pub fn into_errors(self) -> Vec<E> {
        self.errs
    }

    /// Convert this `ParseResult` into a tuple containing the output, if any existed, and errors,
    /// if any were encountered. This matches the output of the old [`Parser::parse_recovery`].
    pub fn into_output_errors(self) -> (Option<T>, Vec<E>) {
        (self.output, self.errs)
    }

    /// Convert this `ParseResult` into a standard `Result`. This discards output if parsing
    /// generated any errors, matching the old behavior of [`Parser::parse`].
    pub fn into_result(self) -> Result<T, Vec<E>> {
        if self.errs.is_empty() {
            self.output.ok_or(self.errs)
        } else {
            Err(self.errs)
        }
    }
}

#[doc(hidden)]
pub struct Located<E> {
    pos: usize,
    err: E,
}

impl<E> Located<E> {
    pub fn at(pos: usize, err: E) -> Self {
        Self { pos, err }
    }

    fn at_pos(pos: usize, err: E) -> Self {
        Self { pos, err }
    }

    fn prioritize(self, other: Self, merge: impl FnOnce(E, E) -> E) -> Self {
        match self.pos.cmp(&other.pos) {
            Ordering::Equal => Self::at_pos(self.pos, merge(self.err, other.err)),
            Ordering::Greater => self,
            Ordering::Less => other,
        }
    }
}

mod internal {
    use super::*;

    pub trait ModeSealed {}

    impl ModeSealed for Emit {}
    impl ModeSealed for Check {}

    /// An abstract parse mode - can be [`Emit`] or [`Check`] in practice, and represents the
    /// common interface for handling both in the same method.
    pub trait Mode: ModeSealed {
        /// The output of this mode for a given type
        type Output<T>;

        /// Bind the result of a closure into an output
        fn bind<T, F: FnOnce() -> T>(f: F) -> Self::Output<T>;

        /// Given an [`Output`](Self::Output), takes its value and return a newly generated output
        fn map<T, U, F: FnOnce(T) -> U>(x: Self::Output<T>, f: F) -> Self::Output<U>;

        /// Given two [`Output`](Self::Output)s, take their values and combine them into a new
        /// output value
        fn combine<T, U, V, F: FnOnce(T, U) -> V>(
            x: Self::Output<T>,
            y: Self::Output<U>,
            f: F,
        ) -> Self::Output<V>;

        /// Given an array of outputs, bind them into an output of arrays
        fn array<T, const N: usize>(x: [Self::Output<T>; N]) -> Self::Output<[T; N]>;

        /// Invoke a parser user the current mode. This is normally equivalent to
        /// [`parser.go::<M>(inp)`](Parser::go), but it can be called on unsized values such as
        /// `dyn Parser`.
        fn invoke<'a, I, O, E, P>(
            parser: &P,
            inp: &mut InputRef<'a, '_, I, E>,
        ) -> PResult<Self, O, E::Error>
        where
            I: Input + ?Sized,
            E: ParserExtra<'a, I>,
            P: Parser<'a, I, O, E> + ?Sized;

        /// Invoke a parser with configuration using the current mode. This is normally equivalent
        /// to [`parser.go::<M>(inp)`](Parser::go_cfg), but it can be called on unsized values
        /// such as `dyn Parser`.
        fn invoke_cfg<'a, I, O, E, P>(
            parser: &P,
            inp: &mut InputRef<'a, '_, I, E>,
            cfg: P::Config,
        ) -> PResult<Self, O, E::Error>
        where
            I: Input + ?Sized,
            E: ParserExtra<'a, I>,
            P: ConfigParser<'a, I, O, E> + ?Sized;
    }

    /// Emit mode - generates parser output
    pub struct Emit;

    impl Mode for Emit {
        type Output<T> = T;
        #[inline]
        fn bind<T, F: FnOnce() -> T>(f: F) -> Self::Output<T> {
            f()
        }
        #[inline]
        fn map<T, U, F: FnOnce(T) -> U>(x: Self::Output<T>, f: F) -> Self::Output<U> {
            f(x)
        }
        #[inline]
        fn combine<T, U, V, F: FnOnce(T, U) -> V>(
            x: Self::Output<T>,
            y: Self::Output<U>,
            f: F,
        ) -> Self::Output<V> {
            f(x, y)
        }
        #[inline]
        fn array<T, const N: usize>(x: [Self::Output<T>; N]) -> Self::Output<[T; N]> {
            x
        }

        #[inline]
        fn invoke<'a, I, O, E, P>(
            parser: &P,
            inp: &mut InputRef<'a, '_, I, E>,
        ) -> PResult<Self, O, E::Error>
        where
            I: Input + ?Sized,
            E: ParserExtra<'a, I>,
            P: Parser<'a, I, O, E> + ?Sized,
        {
            parser.go_emit(inp)
        }

        #[inline]
        fn invoke_cfg<'a, I, O, E, P>(
            parser: &P,
            inp: &mut InputRef<'a, '_, I, E>,
            cfg: P::Config,
        ) -> PResult<Self, O, E::Error>
        where
            I: Input + ?Sized,
            E: ParserExtra<'a, I>,
            P: ConfigParser<'a, I, O, E> + ?Sized,
        {
            parser.go_emit_cfg(inp, cfg)
        }
    }

    /// Check mode - all output is discarded, and only uses parsers to check validity
    pub struct Check;

    impl Mode for Check {
        type Output<T> = ();
        #[inline]
        fn bind<T, F: FnOnce() -> T>(_: F) -> Self::Output<T> {}
        #[inline]
        fn map<T, U, F: FnOnce(T) -> U>(_: Self::Output<T>, _: F) -> Self::Output<U> {}
        #[inline]
        fn combine<T, U, V, F: FnOnce(T, U) -> V>(
            _: Self::Output<T>,
            _: Self::Output<U>,
            _: F,
        ) -> Self::Output<V> {
        }
        #[inline]
        fn array<T, const N: usize>(_: [Self::Output<T>; N]) -> Self::Output<[T; N]> {}

        #[inline]
        fn invoke<'a, I, O, E, P>(
            parser: &P,
            inp: &mut InputRef<'a, '_, I, E>,
        ) -> PResult<Self, O, E::Error>
        where
            I: Input + ?Sized,
            E: ParserExtra<'a, I>,
            P: Parser<'a, I, O, E> + ?Sized,
        {
            parser.go_check(inp)
        }

        #[inline]
        fn invoke_cfg<'a, I, O, E, P>(
            parser: &P,
            inp: &mut InputRef<'a, '_, I, E>,
            cfg: P::Config,
        ) -> PResult<Self, O, E::Error>
        where
            I: Input + ?Sized,
            E: ParserExtra<'a, I>,
            P: ConfigParser<'a, I, O, E> + ?Sized,
        {
            parser.go_check_cfg(inp, cfg)
        }
    }
}

pub use internal::{Check, Emit, Mode};

/// A trait implemented by parsers.
///
/// Parsers take a stream of tokens of type `I` and attempt to parse them into a value of type `O`. In doing so, they
/// may encounter errors. These need not be fatal to the parsing process: syntactic errors can be recovered from and a
/// valid output may still be generated alongside any syntax errors that were encountered along the way. Usually, this
/// output comes in the form of an [Abstract Syntax Tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree) (AST).
///
/// You should not need to implement this trait by hand. If you cannot combine existing combintors (and in particular
/// [`custom`]) to create the combinator you're looking for, please
/// [open an issue](https://github.com/zesterer/chumsky/issues/new)! If you *really* need to implement this trait,
/// please check the documentation in the source: some implementation details have been deliberately hidden.
#[cfg_attr(
    feature = "nightly",
    rustc_on_unimplemented(
        message = "`{Self}` is not a parser from `{I}` to `{O}`",
        label = "This parser is not compatible because it does not implement `Parser<{I}, {O}>`",
        note = "You should check that the output types of your parsers are consistent with combinator you're using",
    )
)]
pub trait Parser<'a, I: Input + ?Sized, O, E: ParserExtra<'a, I> = extra::Default> {
    /// Parse a stream of tokens, yielding an output if possible, and any errors encountered along the way.
    ///
    /// If `None` is returned (i.e: parsing failed) then there will *always* be at least one item in the error `Vec`.
    /// If you want to include non-default state, use [`Parser::parse_with_state`] instead.
    ///
    /// Although the signature of this function looks complicated, it's simpler than you think! You can pass a
    /// [`&[I]`], a [`&str`], or anything implementing [`Stream`] to it.
    fn parse(&self, input: &'a I) -> ParseResult<O, E::Error>
    where
        Self: Sized,
        E::State: Default,
        E::Context: Default,
    {
        self.parse_with_state(input, &mut E::State::default())
    }

    /// Parse a stream of tokens, yielding an output if possible, and any errors encountered along the way.
    /// The provided state will be passed on to parsers that expect it, such as [`map_with_state`](Parser::map_with_state).
    ///
    /// If `None` is returned (i.e: parsing failed) then there will *always* be at least one item in the error `Vec`.
    /// If you want to just use a default state value, use [`Parser::parse`] instead.
    ///
    /// Although the signature of this function looks complicated, it's simpler than you think! You can pass a
    /// [`&[I]`], a [`&str`], or anything implementing [`Stream`] to it.
    fn parse_with_state(&self, input: &'a I, state: &mut E::State) -> ParseResult<O, E::Error>
    where
        Self: Sized,
        E::Context: Default,
    {
        let mut inp = InputRef::new(input, Ok(state));
        let res = self.go::<Emit>(&mut inp);
        let mut errs = inp.into_errs();
        let out = match res {
            Ok(out) => Some(out),
            Err(e) => {
                errs.push(e.err);
                None
            }
        };
        ParseResult::new(out, errs)
    }

    /// Parse a stream of tokens, ignoring any output, and returning any errors encountered along the way.
    ///
    /// If parsing failed, then there will *always* be at least one item in the returned `Vec`.
    /// If you want to include non-default state, use [`Parser::check_with_state`] instead.
    ///
    /// Although the signature of this function looks complicated, it's simpler than you think! You can pass a
    /// [`&[I]`], a [`&str`], or anything implementing [`Stream`] to it.
    fn check(&self, input: &'a I) -> ParseResult<(), E::Error>
    where
        Self: Sized,
        E::State: Default,
        E::Context: Default,
    {
        self.check_with_state(input, &mut E::State::default())
    }

    /// Parse a stream of tokens, ignoring any output, and returning any errors encountered along the way.
    ///
    /// If parsing failed, then there will *always* be at least one item in the returned `Vec`.
    /// If you want to just use a default state value, use [`Parser::check`] instead.
    ///
    /// Although the signature of this function looks complicated, it's simpler than you think! You can pass a
    /// [`&[I]`], a [`&str`], or anything implementing [`Stream`] to it.
    fn check_with_state(&self, input: &'a I, state: &mut E::State) -> ParseResult<(), E::Error>
    where
        Self: Sized,
        E::Context: Default,
    {
        let mut inp = InputRef::new(input, Ok(state));
        let res = self.go::<Check>(&mut inp);
        let mut errs = inp.into_errs();
        let out = match res {
            Ok(_) => Some(()),
            Err(e) => {
                errs.push(e.err);
                None
            }
        };
        ParseResult::new(out, errs)
    }

    /// Parse a stream with all the bells & whistles. You can use this to implement your own parser combinators. Note
    /// that both the signature and semantic requirements of this function are very likely to change in later versions.
    /// Where possible, prefer more ergonomic combinators provided elsewhere in the crate rather than implementing your
    /// own.
    fn go<M: Mode>(&self, inp: &mut InputRef<'a, '_, I, E>) -> PResult<M, O, E::Error>
    where
        Self: Sized;

    #[doc(hidden)]
    fn go_emit(&self, inp: &mut InputRef<'a, '_, I, E>) -> PResult<Emit, O, E::Error>;
    #[doc(hidden)]
    fn go_check(&self, inp: &mut InputRef<'a, '_, I, E>) -> PResult<Check, O, E::Error>;

    /// Map from a slice of the input based on the current parser's span to a value.
    ///
    /// The returned value may borrow data from the input slice, making this function very useful
    /// for creating zero-copy AST output values
    fn map_slice<U, F: Fn(&'a I::Slice) -> U>(self, f: F) -> MapSlice<'a, Self, I, O, E, F, U>
    where
        Self: Sized,
        I: SliceInput,
        I::Slice: 'a,
    {
        MapSlice {
            parser: self,
            mapper: f,
            phantom: PhantomData,
        }
    }

    /// Convert the output of this parser into a slice of the input, based on the current parser's
    /// span.
    ///
    /// This is effectively a special case of [`map_slice`](Parser::map_slice)`(|x| x)`
    fn slice(self) -> Slice<Self, O>
    where
        Self: Sized,
    {
        Slice {
            parser: self,
            phantom: PhantomData,
        }
    }

    /// Filter the output of this parser, accepting only inputs that match the given predicate.
    ///
    /// The output type of this parser is `I`, the input that was found.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let lowercase = any::<_, extra::Err<Simple<str>>>()
    ///     .filter(char::is_ascii_lowercase)
    ///     .repeated()
    ///     .at_least(1)
    ///     .collect::<String>()
    ///     .then_ignore(end());
    ///
    /// assert_eq!(lowercase.parse("hello").into_result(), Ok("hello".to_string()));
    /// assert!(lowercase.parse("Hello").has_errors());
    /// ```
    fn filter<F: Fn(&O) -> bool>(self, f: F) -> Filter<Self, F>
    where
        Self: Sized,
    {
        Filter {
            parser: self,
            filter: f,
        }
    }

    /// Map the output of this parser to another value.
    ///
    /// The output type of this parser is `U`, the same as the function's output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// #[derive(Debug, PartialEq)]
    /// enum Token { Word(String), Num(u64) }
    ///
    /// let word = any::<_, extra::Err<Simple<str>>>()
    ///     .filter(|c: &char| c.is_alphabetic())
    ///     .repeated().at_least(1)
    ///     .collect::<String>()
    ///     .map(Token::Word);
    ///
    /// let num = any::<_, extra::Err<Simple<str>>>()
    ///     .filter(|c: &char| c.is_ascii_digit())
    ///     .repeated().at_least(1)
    ///     .collect::<String>()
    ///     .map(|s| Token::Num(s.parse().unwrap()));
    ///
    /// let token = word.or(num);
    ///
    /// assert_eq!(token.parse("test").into_result(), Ok(Token::Word("test".to_string())));
    /// assert_eq!(token.parse("42").into_result(), Ok(Token::Num(42)));
    /// ```
    fn map<U, F: Fn(O) -> U>(self, f: F) -> Map<Self, O, F>
    where
        Self: Sized,
    {
        Map {
            parser: self,
            mapper: f,
            phantom: PhantomData,
        }
    }

    /// Map the output of this parser to another value, making use of the pattern's span when doing so.
    ///
    /// This is very useful when generating an AST that attaches a span to each AST node.
    ///
    /// The output type of this parser is `U`, the same as the function's output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::prelude::*;
    /// use std::ops::Range;
    ///
    /// // It's common for AST nodes to use a wrapper type that allows attaching span information to them
    /// #[derive(Debug, PartialEq)]
    /// pub struct Spanned<T>(T, SimpleSpan<usize>);
    ///
    /// let ident = text::ident::<_, _, extra::Err<Simple<str>>>()
    ///     .map_with_span(|ident, span| Spanned(ident, span))
    ///     .padded();
    ///
    /// assert_eq!(ident.parse("hello").into_result(), Ok(Spanned("hello", (0..5).into())));
    /// assert_eq!(ident.parse("       hello   ").into_result(), Ok(Spanned("hello", (7..12).into())));
    /// ```
    fn map_with_span<U, F: Fn(O, I::Span) -> U>(self, f: F) -> MapWithSpan<Self, O, F>
    where
        Self: Sized,
    {
        MapWithSpan {
            parser: self,
            mapper: f,
            phantom: PhantomData,
        }
    }

    /// Map the output of this parser to another value, making use of the parser's state when doing so.
    ///
    /// This is very useful for parsing non context-free grammars.
    ///
    /// The output type of this parser is `U`, the same as the function's output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::prelude::*;
    /// use std::ops::Range;
    ///
    /// // It's common for AST nodes to use a wrapper type that allows attaching span information to them
    /// #[derive(Debug, PartialEq)]
    /// pub struct Spanned<T>(T, SimpleSpan<usize>);
    ///
    /// let ident = text::ident::<_, _, extra::Err<Simple<str>>>()
    ///     .map_with_span(|ident, span| Spanned(ident, span))
    ///     .padded();
    ///
    /// assert_eq!(ident.parse("hello").into_result(), Ok(Spanned("hello", (0..5).into())));
    /// assert_eq!(ident.parse("       hello   ").into_result(), Ok(Spanned("hello", (7..12).into())));
    /// ```
    fn map_with_state<U, F: Fn(O, I::Span, &mut E::State) -> U>(
        self,
        f: F,
    ) -> MapWithState<Self, O, F>
    where
        Self: Sized,
    {
        MapWithState {
            parser: self,
            mapper: f,
            phantom: PhantomData,
        }
    }

    /// After a successful parse, apply a fallible function to the output. If the function produces an error, treat it
    /// as a parsing error.
    ///
    /// If you wish parsing of this pattern to continue when an error is generated instead of halting, consider using
    /// [`Parser::validate`] instead.
    ///
    /// The output type of this parser is `U`, the [`Ok`] return value of the function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::prelude::*;
    /// let byte = text::int::<_, _, extra::Err<Rich<str>>>(10)
    ///     .try_map(|s, span| s
    ///         .parse::<u8>()
    ///         .map_err(|e| Rich::custom(span, e)));
    ///
    /// assert!(byte.parse("255").has_output());
    /// assert!(byte.parse("256").has_errors()); // Out of range
    /// ```
    #[doc(alias = "filter_map")]
    fn try_map<U, F: Fn(O, I::Span) -> Result<U, E::Error>>(self, f: F) -> TryMap<Self, O, F>
    where
        Self: Sized,
    {
        TryMap {
            parser: self,
            mapper: f,
            phantom: PhantomData,
        }
    }

    /// After a successful parse, apply a fallible function to the output, making use of the parser's state when
    /// doing so. If the function produces an error, treat it as a parsing error.
    ///
    /// If you wish parsing of this pattern to continue when an error is generated instead of halting, consider using
    /// [`Parser::validate`] instead.
    ///
    /// The output type of this parser is `U`, the [`Ok`] return value of the function.
    fn try_map_with_state<U, F: Fn(O, I::Span, &mut E::State) -> Result<U, E::Error>>(
        self,
        f: F,
    ) -> TryMapWithState<Self, O, F>
    where
        Self: Sized,
    {
        TryMapWithState {
            parser: self,
            mapper: f,
            phantom: PhantomData,
        }
    }

    /// Ignore the output of this parser, yielding `()` as an output instead.
    ///
    /// This can be used to reduce the cost of parsing by avoiding unnecessary allocations (most collections containing
    /// [ZSTs](https://doc.rust-lang.org/nomicon/exotic-sizes.html#zero-sized-types-zsts)
    /// [do not allocate](https://doc.rust-lang.org/std/vec/struct.Vec.html#guarantees)). For example, it's common to
    /// want to ignore whitespace in many grammars (see [`text::whitespace`]).
    ///
    /// The output type of this parser is `()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// // A parser that parses any number of whitespace characters without allocating
    /// let whitespace = any::<_, extra::Err<Simple<str>>>()
    ///     .filter(|c: &char| c.is_whitespace())
    ///     .ignored()
    ///     .repeated()
    ///     .collect::<Vec<_>>();
    ///
    /// assert_eq!(whitespace.parse("    ").into_result(), Ok(vec![(); 4]));
    /// assert_eq!(whitespace.parse("  hello").into_result(), Ok(vec![(); 2]));
    /// ```
    fn ignored(self) -> Ignored<Self, O>
    where
        Self: Sized,
    {
        Ignored {
            parser: self,
            phantom: PhantomData,
        }
    }

    /// Transform all outputs of this parser to a pretermined value.
    ///
    /// The output type of this parser is `U`, the type of the predetermined value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// #[derive(Clone, Debug, PartialEq)]
    /// enum Op { Add, Sub, Mul, Div }
    ///
    /// let op = just::<_, _, extra::Err<Simple<str>>>('+').to(Op::Add)
    ///     .or(just('-').to(Op::Sub))
    ///     .or(just('*').to(Op::Mul))
    ///     .or(just('/').to(Op::Div));
    ///
    /// assert_eq!(op.parse("+").into_result(), Ok(Op::Add));
    /// assert_eq!(op.parse("/").into_result(), Ok(Op::Div));
    /// ```
    fn to<U: Clone>(self, to: U) -> To<Self, O, U, E>
    where
        Self: Sized,
    {
        To {
            parser: self,
            to,
            phantom: PhantomData,
        }
    }

    /// Parse one thing and then another thing, yielding a tuple of the two outputs.
    ///
    /// The output type of this parser is `(O, U)`, a combination of the outputs of both parsers.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let word = any::<_, extra::Err<Simple<str>>>()
    ///     .filter(|c: &char| c.is_alphabetic())
    ///     .repeated()
    ///     .at_least(1)
    ///     .collect::<String>();
    /// let two_words = word.then_ignore(just(' ')).then(word);
    ///
    /// assert_eq!(two_words.parse("dog cat").into_result(), Ok(("dog".to_string(), "cat".to_string())));
    /// assert!(two_words.parse("hedgehog").has_errors());
    /// ```
    fn then<U, B: Parser<'a, I, U, E>>(self, other: B) -> Then<Self, B, O, U, E>
    where
        Self: Sized,
    {
        Then {
            parser_a: self,
            parser_b: other,
            phantom: PhantomData,
        }
    }

    /// Parse one thing and then another thing, yielding only the output of the latter.
    ///
    /// The output type of this parser is `U`, the same as the second parser.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let zeroes = any::<_, extra::Err<Simple<str>>>().filter(|c: &char| *c == '0').ignored().repeated().collect::<Vec<_>>();
    /// let digits = any().filter(|c: &char| c.is_ascii_digit())
    ///     .repeated()
    ///     .collect::<String>();
    /// let integer = zeroes
    ///     .ignore_then(digits)
    ///     .from_str()
    ///     .unwrapped();
    ///
    /// assert_eq!(integer.parse("00064").into_result(), Ok(64));
    /// assert_eq!(integer.parse("32").into_result(), Ok(32));
    /// ```
    fn ignore_then<U, B: Parser<'a, I, U, E>>(self, other: B) -> IgnoreThen<Self, B, O, E>
    where
        Self: Sized,
    {
        IgnoreThen {
            parser_a: self,
            parser_b: other,
            phantom: PhantomData,
        }
    }

    /// Parse one thing and then another thing, yielding only the output of the former.
    ///
    /// The output type of this parser is `O`, the same as the original parser.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let word = any::<_, extra::Err<Simple<str>>>()
    ///     .filter(|c: &char| c.is_alphabetic())
    ///     .repeated()
    ///     .at_least(1)
    ///     .collect::<String>();
    ///
    /// let punctuated = word
    ///     .then_ignore(just('!').or(just('?')).or_not());
    ///
    /// let sentence = punctuated
    ///     .padded() // Allow for whitespace gaps
    ///     .repeated()
    ///     .collect::<Vec<_>>();
    ///
    /// assert_eq!(
    ///     sentence.parse("hello! how are you?").into_result(),
    ///     Ok(vec![
    ///         "hello".to_string(),
    ///         "how".to_string(),
    ///         "are".to_string(),
    ///         "you".to_string(),
    ///     ]),
    /// );
    /// ```
    fn then_ignore<U, B: Parser<'a, I, U, E>>(self, other: B) -> ThenIgnore<Self, B, U, E>
    where
        Self: Sized,
    {
        ThenIgnore {
            parser_a: self,
            parser_b: other,
            phantom: PhantomData,
        }
    }

    /// Parse one thing and then another thing, creating the second parser from the result of
    /// the first. If you only have a couple cases to handle, prefer [`Parser::or`].
    ///
    /// The output of this parser is `U`, the result of the second parser
    ///
    /// Error recovery for this parser may be sub-optimal, as if the first parser succeeds on
    /// recovery then the second produces an error, the primary error will point to the location in
    /// the second parser which failed, ignoring that the first parser may be the root cause. There
    /// may be other pathological errors cases as well.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// // A parser that parses a single letter and then its successor
    /// let successive_letters = one_of::<_, _, extra::Err<Simple<[u8]>>>((b'a'..=b'z').collect::<Vec<u8>>())
    ///     .then_with(|letter: u8| just(letter + 1));
    ///
    /// assert_eq!(successive_letters.parse(b"ab").into_result(), Ok(b'b')); // 'b' follows 'a'
    /// assert!(successive_letters.parse(b"ac").has_errors()); // 'c' does not follow 'a'
    /// ```
    fn then_with<U, B: Parser<'a, I, U, E>, F: Fn(O) -> B>(
        self,
        then: F,
    ) -> ThenWith<Self, B, O, F, I, E>
    where
        Self: Sized,
    {
        ThenWith {
            parser: self,
            then,
            phantom: PhantomData,
        }
    }

    /// Parse one thing and then another thing, creating the second parser from the result of
    /// the first. If you only have a couple cases to handle, prefer [`Parser::or`].
    ///
    /// The output of this parser is `U`, the result of the second parser
    ///
    /// Error recovery for this parser may be sub-optimal, as if the first parser succeeds on
    /// recovery then the second produces an error, the primary error will point to the location in
    /// the second parser which failed, ignoring that the first parser may be the root cause. There
    /// may be other pathological errors cases as well.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let successor = just(b'\0').configure(|cfg, ctx: &u8| cfg.seq(*ctx + 1));
    ///
    /// // A parser that parses a single letter and then its successor
    /// let successive_letters = one_of::<_, _, extra::Err<Simple<[u8]>>>(b'a'..=b'z')
    ///     .then_with_ctx(successor);
    ///
    /// assert_eq!(successive_letters.parse(b"ab").into_result(), Ok(b'b')); // 'b' follows 'a'
    /// assert!(successive_letters.parse(b"ac").has_errors()); // 'c' does not follow 'a'
    /// ```
    fn then_with_ctx<U, P>(
        self,
        then: P,
    ) -> ThenWithCtx<Self, P, O, I, extra::Full<E::Error, E::State, O>>
    where
        Self: Sized,
        O: 'a,
        P: Parser<'a, I, U, extra::Full<E::Error, E::State, O>>,
    {
        ThenWithCtx {
            parser: self,
            then,
            phantom: PhantomData,
        }
    }

    /// Run the previous contextual parser with the provided context
    ///
    /// ```
    /// # use chumsky::zero_copy::prelude::*;
    /// # use chumsky::zero_copy::primitive::JustCfg;
    ///
    /// let generic = just(b'0').configure(|cfg, ctx: &u8| cfg.seq(*ctx));
    ///
    /// let parse_a = just::<_, _, extra::Default>(b'b').ignore_then(generic.with_ctx::<u8>(b'a'));
    /// let parse_b = just::<_, _, extra::Default>(b'a').ignore_then(generic.with_ctx(b'b'));
    ///
    /// assert_eq!(parse_a.parse(b"ba" as &[_]).into_result(), Ok::<_, Vec<EmptyErr>>(b'a'));
    /// assert!(parse_a.parse(b"bb").has_errors());
    /// assert_eq!(parse_b.parse(b"ab" as &[_]).into_result(), Ok(b'b'));
    /// assert!(parse_b.parse(b"aa").has_errors());
    /// ```
    fn with_ctx<Ctx>(self, ctx: Ctx) -> WithCtx<Self, Ctx>
    where
        Self: Sized,
        Ctx: 'a + Clone,
    {
        WithCtx { parser: self, ctx }
    }

    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    ///
    /// // Lua-style multiline string literal
    /// let string = just::<_, _, extra::Err<Simple<str>>>('=')
    ///     .repeated()
    ///     .map_slice(str::len)
    ///     .padded_by(just('['))
    ///     .then_with(|n| {
    ///         let close = just('=').repeated().exactly(n).padded_by(just(']'));
    ///         any()
    ///             .and_is(close.not())
    ///             .repeated()
    ///             .slice()
    ///             .then_ignore(close)
    ///     });
    ///
    /// assert_eq!(
    ///     string.parse("[[wxyz]]").into_result(),
    ///     Ok("wxyz"),
    /// );
    /// assert_eq!(
    ///     string.parse("[==[abcd]=]efgh]===]ijkl]==]").into_result(),
    ///     Ok("abcd]=]efgh]===]ijkl"),
    /// );
    /// ```
    fn and_is<U, B>(self, other: B) -> AndIs<Self, B, U>
    where
        Self: Sized,
        B: Parser<'a, I, U, E>,
    {
        AndIs {
            parser_a: self,
            parser_b: other,
            phantom: PhantomData,
        }
    }

    /// Parse the pattern surrounded by the given delimiters.
    ///
    /// The output type of this parser is `O`, the same as the original parser.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// // A LISP-style S-expression
    /// #[derive(Debug, PartialEq)]
    /// enum SExpr {
    ///     Ident(String),
    ///     Num(u64),
    ///     List(Vec<SExpr>),
    /// }
    ///
    /// let ident = any::<_, extra::Err<Simple<str>>>().filter(|c: &char| c.is_alphabetic())
    ///     .repeated()
    ///     .at_least(1)
    ///     .collect::<String>();
    ///
    /// let num = text::int(10)
    ///     .from_str()
    ///     .unwrapped();
    ///
    /// let s_expr = recursive(|s_expr| s_expr
    ///     .padded()
    ///     .repeated()
    ///     .collect::<Vec<_>>()
    ///     .map(SExpr::List)
    ///     .delimited_by(just('('), just(')'))
    ///     .or(ident.map(SExpr::Ident))
    ///     .or(num.map(SExpr::Num)));
    ///
    /// // A valid input
    /// assert_eq!(
    ///     s_expr.parse("(add (mul 42 3) 15)").into_result(),
    ///     Ok(SExpr::List(vec![
    ///         SExpr::Ident("add".to_string()),
    ///         SExpr::List(vec![
    ///             SExpr::Ident("mul".to_string()),
    ///             SExpr::Num(42),
    ///             SExpr::Num(3),
    ///         ]),
    ///         SExpr::Num(15),
    ///     ])),
    /// );
    /// ```
    fn delimited_by<U, V, B, C>(self, start: B, end: C) -> DelimitedBy<Self, B, C, U, V>
    where
        Self: Sized,
        B: Parser<'a, I, U, E>,
        C: Parser<'a, I, V, E>,
    {
        DelimitedBy {
            parser: self,
            start,
            end,
            phantom: PhantomData,
        }
    }

    /// Parse a pattern, but with an instance of another pattern on either end, yielding the output of the inner.
    ///
    /// The output type of this parser is `O`, the same as the original parser.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let ident = text::ident::<_, _, extra::Err<Simple<str>>>()
    ///     .padded_by(just('!'));
    ///
    /// assert_eq!(ident.parse("!hello!").into_result(), Ok("hello"));
    /// assert!(ident.parse("hello!").has_errors());
    /// assert!(ident.parse("!hello").has_errors());
    /// assert!(ident.parse("hello").has_errors());
    /// ```
    fn padded_by<U, B>(self, padding: B) -> PaddedBy<Self, B, U>
    where
        Self: Sized,
        B: Parser<'a, I, U, E>,
    {
        PaddedBy {
            parser: self,
            padding,
            phantom: PhantomData,
        }
    }

    /// Parse one thing or, on failure, another thing.
    ///
    /// The output of both parsers must be of the same type, because either output can be produced.
    ///
    /// If both parser succeed, the output of the first parser is guaranteed to be prioritised over the output of the
    /// second.
    ///
    /// If both parsers produce errors, the combinator will attempt to select from or combine the errors to produce an
    /// error that is most likely to be useful to a human attempting to understand the problem. The exact algorithm
    /// used is left unspecified, and is not part of the crate's semver guarantees, although regressions in error
    /// quality should be reported in the issue tracker of the main repository.
    ///
    /// Please note that long chains of [`Parser::or`] combinators have been known to result in poor compilation times.
    /// If you feel you are experiencing this, consider using [`choice`] instead.
    ///
    /// The output type of this parser is `O`, the output of both parsers.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let op = just::<_, _, extra::Err<Simple<str>>>('+')
    ///     .or(just('-'))
    ///     .or(just('*'))
    ///     .or(just('/'));
    ///
    /// assert_eq!(op.parse("+").into_result(), Ok('+'));
    /// assert_eq!(op.parse("/").into_result(), Ok('/'));
    /// assert!(op.parse("!").has_errors());
    /// ```
    fn or<B>(self, other: B) -> Or<Self, B>
    where
        Self: Sized,
        B: Parser<'a, I, O, E>,
    {
        Or {
            parser_a: self,
            parser_b: other,
        }
    }

    /// Attempt to parse something, but only if it exists.
    ///
    /// If parsing of the pattern is successful, the output is `Some(_)`. Otherwise, the output is `None`.
    ///
    /// The output type of this parser is `Option<O>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let word = any::<_, extra::Err<Simple<str>>>().filter(|c: &char| c.is_alphabetic())
    ///     .repeated()
    ///     .at_least(1)
    ///     .collect::<String>();
    ///
    /// let word_or_question = word
    ///     .then(just('?').or_not());
    ///
    /// assert_eq!(word_or_question.parse("hello?").into_result(), Ok(("hello".to_string(), Some('?'))));
    /// assert_eq!(word_or_question.parse("wednesday").into_result(), Ok(("wednesday".to_string(), None)));
    /// ```
    fn or_not(self) -> OrNot<Self>
    where
        Self: Sized,
    {
        OrNot { parser: self }
    }

    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    ///
    /// #[derive(Debug, PartialEq)]
    /// enum Tree<'a> {
    ///     Text(&'a str),
    ///     Group(Vec<Self>),
    /// }
    ///
    /// // Arbitrary text, nested in a tree with { ... } delimiters
    /// let tree = recursive::<_, _, extra::Err<Simple<str>>, _, _>(|tree| {
    ///     let text = any()
    ///         .and_is(one_of("{}").not())
    ///         .repeated()
    ///         .at_least(1)
    ///         .map_slice(Tree::Text);
    ///
    ///     let group = tree
    ///         .repeated()
    ///         .collect()
    ///         .delimited_by(just('{'), just('}'))
    ///         .map(Tree::Group);
    ///
    ///     text.or(group)
    /// });
    ///
    /// assert_eq!(
    ///     tree.parse("{abcd{efg{hijk}lmn{opq}rs}tuvwxyz}").into_result(),
    ///     Ok(Tree::Group(vec![
    ///         Tree::Text("abcd"),
    ///         Tree::Group(vec![
    ///             Tree::Text("efg"),
    ///             Tree::Group(vec![
    ///                 Tree::Text("hijk"),
    ///             ]),
    ///             Tree::Text("lmn"),
    ///             Tree::Group(vec![
    ///                 Tree::Text("opq"),
    ///             ]),
    ///             Tree::Text("rs"),
    ///         ]),
    ///         Tree::Text("tuvwxyz"),
    ///     ])),
    /// );
    /// ```
    fn not(self) -> Not<Self, O>
    where
        Self: Sized,
    {
        Not {
            parser: self,
            phantom: PhantomData,
        }
    }

    /// Parse a pattern any number of times (including zero times).
    ///
    /// Input is eagerly parsed. Be aware that the parser will accept no occurences of the pattern too. Consider using
    /// [`Repeated::at_least`] instead if it better suits your use-case.
    ///
    /// The output type of this parser can be any [`Container`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let num = any::<_, extra::Err<Simple<str>>>()
    ///     .filter(|c: &char| c.is_ascii_digit())
    ///     .repeated()
    ///     .at_least(1)
    ///     .collect::<String>()
    ///     .from_str()
    ///     .unwrapped();
    ///
    /// let sum = num.clone().then(just('+').ignore_then(num).repeated().collect::<Vec<_>>())
    ///     .foldl(|a, b| a + b);
    ///
    /// assert_eq!(sum.parse("2+13+4+0+5").into_result(), Ok(24));
    /// ```
    fn repeated(self) -> Repeated<Self, O, I, E>
    where
        Self: Sized,
    {
        Repeated {
            parser: self,
            at_least: 0,
            at_most: !0,
            phantom: PhantomData,
        }
    }

    /// Parse a pattern a specific number of times.
    ///
    /// Input is eagerly parsed. Consider using [`RepeatedExactly::repeated`] if a non-constant number of values are expected.
    ///
    /// The output type of this parser can be any [`ContainerExactly`].
    fn repeated_exactly<const N: usize>(self) -> RepeatedExactly<Self, O, (), N>
    where
        Self: Sized,
    {
        RepeatedExactly {
            parser: self,
            phantom: PhantomData,
        }
    }

    /// Parse a pattern, separated by another, any number of times.
    ///
    /// You can use [`SeparatedBy::allow_leading`] or [`SeparatedBy::allow_trailing`] to allow leading or trailing
    /// separators.
    ///
    /// The output type of this parser can be any [`Container`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let shopping = text::ident::<_, _, extra::Err<Simple<str>>>()
    ///     .padded()
    ///     .separated_by(just(','))
    ///     .collect::<Vec<_>>();
    ///
    /// assert_eq!(shopping.parse("eggs").into_result(), Ok(vec!["eggs"]));
    /// assert_eq!(shopping.parse("eggs, flour, milk").into_result(), Ok(vec!["eggs", "flour", "milk"]));
    /// ```
    ///
    /// See [`SeparatedBy::allow_leading`] and [`SeparatedBy::allow_trailing`] for more examples.
    fn separated_by<U, B>(self, separator: B) -> SeparatedBy<Self, B, O, U, I, E>
    where
        Self: Sized,
        B: Parser<'a, I, U, E>,
    {
        SeparatedBy {
            parser: self,
            separator,
            at_least: 0,
            at_most: !0,
            allow_leading: false,
            allow_trailing: false,
            phantom: PhantomData,
        }
    }

    /// Parse a pattern, separated by another, a specific number of times.
    ///
    /// You can use [`SeparatedByExactly::allow_leading`] or [`SeparatedByExactly::allow_trailing`] to
    /// allow leading or trailing separators.
    ///
    /// The output type of this parser can be any [`ContainerExactly`].
    fn separated_by_exactly<U, B, const N: usize>(
        self,
        separator: B,
    ) -> SeparatedByExactly<Self, B, U, (), N>
    where
        Self: Sized,
        B: Parser<'a, I, U, E>,
    {
        SeparatedByExactly {
            parser: self,
            separator,
            allow_leading: false,
            allow_trailing: false,
            phantom: PhantomData,
        }
    }

    /// Right-fold the output of the parser into a single value.
    ///
    /// The output of the original parser must be of type `(impl IntoIterator<Item = A>, B)`. Because right-folds work
    /// backwards, the iterator must implement [`DoubleEndedIterator`] so that it can be reversed.
    ///
    /// The output type of this parser is `B`, the right-hand component of the original parser's output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let int = text::int::<_, _, extra::Err<Simple<str>>>(10)
    ///     .from_str()
    ///     .unwrapped();
    ///
    /// let signed = just('+').to(1)
    ///     .or(just('-').to(-1))
    ///     .repeated()
    ///     .collect::<Vec<_>>()
    ///     .then(int)
    ///     .foldr(|a, b| a * b);
    ///
    /// assert_eq!(signed.parse("3").into_result(), Ok(3));
    /// assert_eq!(signed.parse("-17").into_result(), Ok(-17));
    /// assert_eq!(signed.parse("--+-+-5").into_result(), Ok(5));
    /// ```
    fn foldr<A, B, F>(self, f: F) -> Foldr<Self, F, A, B, E>
    where
        Self: Parser<'a, I, (A, B), E> + Sized,
        A: IntoIterator,
        A::IntoIter: DoubleEndedIterator,
        F: Fn(A::Item, B) -> B,
    {
        Foldr {
            parser: self,
            folder: f,
            phantom: PhantomData,
        }
    }

    /// Left-fold the output of the parser into a single value.
    ///
    /// The output of the original parser must be of type `(A, impl IntoIterator<Item = B>)`.
    ///
    /// The output type of this parser is `A`, the left-hand component of the original parser's output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let int = text::int::<_, _, extra::Err<Simple<str>>>(10)
    ///     .from_str()
    ///     .unwrapped();
    ///
    /// let sum = int
    ///     .clone()
    ///     .then(just('+').ignore_then(int).repeated().collect::<Vec<_>>())
    ///     .foldl(|a, b| a + b);
    ///
    /// assert_eq!(sum.parse("1+12+3+9").into_result(), Ok(25));
    /// assert_eq!(sum.parse("6").into_result(), Ok(6));
    /// ```
    fn foldl<A, B, F>(self, f: F) -> Foldl<Self, F, A, B, E>
    where
        Self: Parser<'a, I, (A, B), E> + Sized,
        B: IntoIterator,
        F: Fn(A, B::Item) -> A,
    {
        Foldl {
            parser: self,
            folder: f,
            phantom: PhantomData,
        }
    }

    /// Parse a pattern. Afterwards, the input stream will be rewound to its original state, as if parsing had not
    /// occurred.
    ///
    /// This combinator is useful for cases in which you wish to avoid a parser accidentally consuming too much input,
    /// causing later parsers to fail as a result. A typical use-case of this is that you want to parse something that
    /// is not followed by something else.
    ///
    /// The output type of this parser is `O`, the same as the original parser.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::prelude::*;
    /// let just_numbers = text::digits::<_, _, extra::Err<Simple<str>>>(10)
    ///     .slice()
    ///     .padded()
    ///     .then_ignore(none_of("+-*/").rewind())
    ///     .separated_by(just(','))
    ///     .collect::<Vec<_>>();
    /// // 3 is not parsed because it's followed by '+'.
    /// assert_eq!(just_numbers.parse("1, 2, 3 + 4").into_result(), Ok(vec!["1", "2"]));
    /// ```
    fn rewind(self) -> Rewind<Self>
    where
        Self: Sized,
    {
        Rewind { parser: self }
    }

    /// Parse a pattern, ignoring any amount of whitespace both before and after the pattern.
    ///
    /// The output type of this parser is `O`, the same as the original parser.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::prelude::*;
    /// let ident = text::ident::<_, _, extra::Err<Simple<str>>>().padded();
    ///
    /// // A pattern with no whitespace surrounding it is accepted
    /// assert_eq!(ident.parse("hello").into_result(), Ok("hello"));
    /// // A pattern with arbitrary whitespace surrounding it is also accepted
    /// assert_eq!(ident.parse(" \t \n  \t   world  \t  ").into_result(), Ok("world"));
    /// ```
    fn padded(self) -> Padded<Self>
    where
        Self: Sized,
        I: Input,
        I::Token: Char,
    {
        Padded { parser: self }
    }

    /// Flatten a nested collection.
    ///
    /// This use-cases of this method are broadly similar to those of [`Iterator::flatten`].
    ///
    /// The output type of this parser is `Vec<T>`, where the original parser output was
    /// `impl IntoIterator<Item = impl IntoIterator<Item = T>>`.
    fn flatten<T, Inner>(self) -> Map<Self, O, fn(O) -> Vec<T>>
    where
        Self: Sized,
        O: IntoIterator<Item = Inner>,
        Inner: IntoIterator<Item = T>,
    {
        self.map(|xs| xs.into_iter().flat_map(|xs| xs.into_iter()).collect())
    }

    /// Apply a fallback recovery strategy to this parser should it fail.
    ///
    /// There is no silver bullet for error recovery, so this function allows you to specify one of several different
    /// strategies at the location of your choice. Prefer an error recovery strategy that more precisely mirrors valid
    /// syntax where possible to make error recovery more reliable.
    ///
    /// Because chumsky is a [PEG](https://en.m.wikipedia.org/wiki/Parsing_expression_grammar) parser, which always
    /// take the first successful parsing route through a grammar, recovering from an error may cause the parser to
    /// erroneously miss alternative valid routes through the grammar that do not generate recoverable errors. If you
    /// run into cases where valid syntax fails to parse without errors, this might be happening: consider removing
    /// error recovery or switching to a more specific error recovery strategy.
    ///
    /// The output type of this parser is `O`, the same as the original parser.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// #[derive(Debug, PartialEq)]
    /// enum Expr<'a> {
    ///     Error,
    ///     Int(&'a str),
    ///     List(Vec<Expr<'a>>),
    /// }
    ///
    /// let recover = just::<_, _, extra::Err<Simple<str>>>('[')
    ///         .ignore_then(take_until(just(']')).ignored());
    ///
    /// let expr = recursive::<_, _, extra::Err<Simple<str>>, _, _>(|expr| expr
    ///     .separated_by(just(','))
    ///     .collect::<Vec<_>>()
    ///     .delimited_by(just('['), just(']'))
    ///     .map(Expr::List)
    ///     // If parsing a list expression fails, recover at the next delimiter, generating an error AST node
    ///     .recover_with(recover.map(|_| Expr::Error))
    ///     .or(text::int(10).map(Expr::Int))
    ///     .padded());
    ///
    /// assert!(expr.parse("five").has_errors()); // Text is not a valid expression in this language...
    /// assert_eq!(
    ///     expr.parse("[1, 2, 3]").into_result(),
    ///     Ok(Expr::List(vec![Expr::Int("1"), Expr::Int("2"), Expr::Int("3")])),
    /// ); // ...but lists and numbers are!
    ///
    /// // This input has two syntax errors...
    /// let res = expr.parse("[[1, two], [3, four]]");
    /// // ...and error recovery allows us to catch both of them!
    /// assert_eq!(res.errors().len(), 2);
    /// // Additionally, the AST we get back still has useful information.
    /// assert_eq!(res.output(), Some(&Expr::List(vec![Expr::Error, Expr::Error])));
    /// ```
    fn recover_with<F: Parser<'a, I, O, E>>(self, fallback: F) -> RecoverWith<Self, F>
    where
        Self: Sized,
    {
        RecoverWith {
            parser: self,
            fallback,
        }
    }

    /// Map the primary error of this parser to another value.
    ///
    /// This function is most useful when using a custom error type, allowing you to augment errors according to
    /// context.
    ///
    /// The output type of this parser is `O`, the same as the original parser.
    // TODO: Map E -> D, not E -> E
    fn map_err<F>(self, f: F) -> MapErr<Self, F>
    where
        Self: Sized,
        F: Fn(E::Error) -> E::Error,
    {
        MapErr {
            parser: self,
            mapper: f,
        }
    }

    /// Map the primary error of this parser to another value, making use of the span from the start of the attempted
    /// to the point at which the error was encountered.
    ///
    /// This function is useful for augmenting errors to allow them to display the span of the initial part of a
    /// pattern, for example to add a "while parsing" clause to your error messages.
    ///
    /// The output type of this parser is `O`, the same as the original parser.
    ///
    // TODO: Map E -> D, not E -> E
    fn map_err_with_span<F>(self, f: F) -> MapErrWithSpan<Self, F>
    where
        Self: Sized,
        F: Fn(E::Error, I::Span) -> E::Error,
    {
        MapErrWithSpan {
            parser: self,
            mapper: f,
        }
    }

    /// Map the primary error of this parser to another value, making use of the parser state.
    ///
    /// This function is useful for augmenting errors to allow them to include context in non context-free
    /// languages, or provide contextual notes on possible causes.
    ///
    /// The output type of this parser is `O`, the same as the original parser.
    ///
    // TODO: Map E -> D, not E -> E
    fn map_err_with_state<F>(self, f: F) -> MapErrWithState<Self, F>
    where
        Self: Sized,
        F: Fn(E::Error, I::Span, &mut E::State) -> E::Error,
    {
        MapErrWithState {
            parser: self,
            mapper: f,
        }
    }

    /// Validate an output, producing non-terminal errors if it does not fulfil certain criteria.
    ///
    /// This function also permits mapping the output to a value of another type, similar to [`Parser::map`].
    ///
    /// If you wish parsing of this pattern to halt when an error is generated instead of continuing, consider using
    /// [`Parser::try_map`] instead.
    ///
    /// The output type of this parser is `U`, the result of the validation closure.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::prelude::*;
    /// let large_int = text::int::<_, _, extra::Err<Rich<str>>>(10)
    ///     .from_str()
    ///     .unwrapped()
    ///     .validate(|x: u32, span, emitter| {
    ///         if x < 256 { emitter.emit(Rich::custom(span, format!("{} must be 256 or higher.", x))) }
    ///         x
    ///     });
    ///
    /// assert_eq!(large_int.parse("537").into_result(), Ok(537));
    /// assert!(large_int.parse("243").into_result().is_err());
    /// ```
    fn validate<U, F>(self, f: F) -> Validate<Self, O, F>
    where
        Self: Sized,
        F: Fn(O, I::Span, &mut Emitter<E::Error>) -> U,
    {
        Validate {
            parser: self,
            validator: f,
            phantom: PhantomData,
        }
    }

    /// Map the primary error of this parser to a result. If the result is [`Ok`], the parser succeeds with that value.
    ///
    /// Note that even if the function returns an [`Ok`], the input stream will still be 'stuck' at the input following
    /// the input that triggered the error. You'll need to follow uses of this combinator with a parser that resets
    /// the input stream to a known-good state (for example, [`take_until`]).
    ///
    /// The output type of this parser is `U`, the [`Ok`] type of the result.
    fn or_else<F>(self, f: F) -> OrElse<Self, F>
    where
        Self: Sized,
        F: Fn(E::Error) -> Result<O, E::Error>,
    {
        OrElse {
            parser: self,
            or_else: f,
        }
    }

    /// Attempt to convert the output of this parser into something else using Rust's [`FromStr`] trait.
    ///
    /// This is most useful when wanting to convert literal values into their corresponding Rust type, such as when
    /// parsing integers.
    ///
    /// The output type of this parser is `Result<U, U::Err>`, the result of attempting to parse the output, `O`, into
    /// the value `U`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::prelude::*;
    /// let uint64 = text::int::<_, _, extra::Err<Simple<str>>>(10)
    ///     .from_str::<u64>()
    ///     .unwrapped();
    ///
    /// assert_eq!(uint64.parse("7").into_result(), Ok(7));
    /// assert_eq!(uint64.parse("42").into_result(), Ok(42));
    /// ```
    #[allow(clippy::wrong_self_convention)]
    fn from_str<U>(self) -> Map<Self, O, fn(O) -> Result<U, U::Err>>
    where
        Self: Sized,
        U: FromStr,
        O: AsRef<str>,
    {
        self.map(|o| o.as_ref().parse())
    }

    /// For parsers that produce a [`Result`] as their output, unwrap the result (panicking if an [`Err`] is
    /// encountered).
    ///
    /// In general, this method should be avoided except in cases where all possible that the parser might produce can
    /// by parsed using [`FromStr`] without producing an error.
    ///
    /// This combinator is not named `unwrap` to avoid confusion: it unwraps *during parsing*, not immediately.
    ///
    /// The output type of this parser is `U`, the [`Ok`] value of the [`Result`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::prelude::*;
    /// let boolean = just::<_, _, extra::Err<Simple<str>>>("true")
    ///     .or(just("false"))
    ///     .from_str::<bool>()
    ///     .unwrapped(); // Cannot panic: the only possible outputs generated by the parser are "true" or "false"
    ///
    /// assert_eq!(boolean.parse("true").into_result(), Ok(true));
    /// assert_eq!(boolean.parse("false").into_result(), Ok(false));
    /// // Does not panic, because the original parser only accepts "true" or "false"
    /// assert!(boolean.parse("42").has_errors());
    /// ```
    fn unwrapped<U, E1>(self) -> Map<Self, Result<U, E1>, fn(Result<U, E1>) -> U>
    where
        Self: Sized + Parser<'a, I, Result<U, E1>, E>,
        E1: fmt::Debug,
    {
        self.map(|o| o.unwrap())
    }

    /// Box the parser, yielding a parser that performs parsing through dynamic dispatch.
    ///
    /// Boxing a parser might be useful for:
    ///
    /// - Dynamically building up parsers at run-time
    ///
    /// - Improving compilation times (Rust can struggle to compile code containing very long types)
    ///
    /// - Passing a parser over an FFI boundary
    ///
    /// - Getting around compiler implementation problems with long types such as
    ///   [this](https://github.com/rust-lang/rust/issues/54540).
    ///
    /// - Places where you need to name the type of a parser
    ///
    /// Boxing a parser is broadly equivalent to boxing other combinators via dynamic dispatch, such as [`Iterator`].
    ///
    /// The output type of this parser is `O`, the same as the original parser.
    fn boxed(self) -> Boxed<'a, I, O, E>
    where
        Self: Sized + 'a,
    {
        Boxed {
            inner: Rc::new(self),
        }
    }

    /// Use Pratt parsing to efficiently parse binary operators
    /// with different associativity.
    ///
    /// The parsing algorithm currently uses recursion
    /// to parse nested expressions.
    ///
    /// # Examples
    ///
    /// ```
    /// use chumsky::zero_copy::prelude::*;
    /// use chumsky::zero_copy::pratt::{InfixOperator, InfixPrecedence, Associativity};
    ///
    /// #[derive(Clone, Copy, Debug)]
    /// enum Operator {
    ///     Add,
    ///     Sub,
    ///     Mul,
    ///     Div,
    /// }
    ///
    /// enum Expr {
    ///     Literal(i64),
    ///     Add(Box<Expr>, Box<Expr>),
    ///     Sub(Box<Expr>, Box<Expr>),
    ///     Mul(Box<Expr>, Box<Expr>),
    ///     Div(Box<Expr>, Box<Expr>),
    /// }
    ///
    /// impl std::fmt::Display for Expr {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         match self {
    ///             Self::Literal(literal) => write!(f, "{literal}"),
    ///             Self::Add(left, right) => write!(f, "({left} + {right})"),
    ///             Self::Sub(left, right) => write!(f, "({left} - {right})"),
    ///             Self::Mul(left, right) => write!(f, "({left} * {right})"),
    ///             Self::Div(left, right) => write!(f, "({left} / {right})"),
    ///         }
    ///     }
    /// }
    ///
    /// impl InfixOperator<Expr> for Operator {
    ///     type Strength = u8;
    ///
    ///     fn precedence(&self) -> InfixPrecedence<Self::Strength> {
    ///         // NOTE: Usually, in Rust for example, all these operators
    ///         // are left-associative. However, in this example we define
    ///         // then with different associativities for demonstration purposes.
    ///         // (Although it doesn't really matter here since these operations
    ///         // are commutative for integers anyway.)
    ///         match self {
    ///             Self::Add => InfixPrecedence::new(0, Associativity::Left),
    ///             Self::Sub => InfixPrecedence::new(0, Associativity::Left),
    ///             Self::Mul => InfixPrecedence::new(1, Associativity::Right),
    ///             Self::Div => InfixPrecedence::new(1, Associativity::Right),
    ///         }
    ///     }
    ///
    ///     fn build_expression(self, left: Expr, right: Expr) -> Expr {
    ///         let (left, right) = (Box::new(left), Box::new(right));
    ///         match self {
    ///             Self::Add => Expr::Add(left, right),
    ///             Self::Sub => Expr::Sub(left, right),
    ///             Self::Mul => Expr::Mul(left, right),
    ///             Self::Div => Expr::Div(left, right),
    ///         }
    ///     }
    /// }
    ///
    /// let atom = text::int::<_, _, extra::Default>(10)
    ///     .from_str()
    ///     .unwrapped()
    ///     .map(Expr::Literal);
    ///
    /// let operator = choice((
    ///     just('+').to(Operator::Add),
    ///     just('-').to(Operator::Sub),
    ///     just('*').to(Operator::Mul),
    ///     just('/').to(Operator::Div),
    /// ));
    ///
    /// let expr = atom.pratt(operator.padded_by(just(' ')));
    /// let expr_str = expr.map(|expr| expr.to_string()).then_ignore(end());
    /// assert_eq!(expr_str.parse("1 + 2").into_result(), Ok("(1 + 2)".to_string()));
    /// // `*` binds more strongly than `+`
    /// assert_eq!(expr_str.parse("1 * 2 + 3").into_result(), Ok("((1 * 2) + 3)".to_string()));
    /// assert_eq!(expr_str.parse("1 + 2 * 3").into_result(), Ok("(1 + (2 * 3))".to_string()));
    /// // `+` is left-associative
    /// assert_eq!(expr_str.parse("1 + 2 + 3").into_result(), Ok("((1 + 2) + 3)".to_string()));
    /// // `*` is right-associative (in this example)
    /// assert_eq!(expr_str.parse("1 * 2 * 3").into_result(), Ok("(1 * (2 * 3))".to_string()));
    /// ```
    fn pratt<OpParser, Op>(self, op_parser: OpParser) -> Pratt<E, Self, O, OpParser, Op>
    where
        Self: Sized,
        OpParser: Parser<'a, I, Op, E>,
        Op: pratt::InfixOperator<O>,
    {
        Pratt {
            parser_atom: self,
            parser_op: op_parser,
            phantom: PhantomData,
        }
    }
}

/// A parser that can be configured with runtime context
pub trait ConfigParser<'a, I, O, E>: Parser<'a, I, O, E>
where
    I: ?Sized + Input,
    E: ParserExtra<'a, I>,
{
    /// Type used to configure the parser
    type Config: Default;

    /// Parse a stream with the provided configured values. This can be used to control a parser's
    /// behavior at parse-time.
    fn go_cfg<M: Mode>(
        &self,
        inp: &mut InputRef<'a, '_, I, E>,
        cfg: Self::Config,
    ) -> PResult<M, O, E::Error>
    where
        Self: Sized;

    #[doc(hidden)]
    fn go_emit_cfg(
        &self,
        inp: &mut InputRef<'a, '_, I, E>,
        cfg: Self::Config,
    ) -> PResult<Emit, O, E::Error>;
    #[doc(hidden)]
    fn go_check_cfg(
        &self,
        inp: &mut InputRef<'a, '_, I, E>,
        cfg: Self::Config,
    ) -> PResult<Check, O, E::Error>;

    /// A combinator that allows configuration of the parser from the current context
    fn configure<F>(self, cfg: F) -> Configure<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Config, &E::Context) -> Self::Config,
    {
        Configure { parser: self, cfg }
    }
}

/// An iterator that wraps an iterable parser. See [`IterParser::parse_iter`].
pub struct ParserIter<
    'a,
    'iter,
    P: IterParser<'a, I, O, E>,
    I: Input + ?Sized,
    O,
    E: ParserExtra<'a, I>,
> {
    parser: P,
    state: P::IterState<Emit>,
    inp: InputRef<'a, 'iter, I, E>,
    phantom: PhantomData<&'a O>,
}

impl<'a, 'iter, P, I: Input + ?Sized, O, E: ParserExtra<'a, I>> Iterator
    for ParserIter<'a, 'iter, P, I, O, E>
where
    P: IterParser<'a, I, O, E>,
{
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        self.parser
            .next::<Emit>(&mut self.inp, &mut self.state)
            .and_then(|res| res.ok())
    }
}

/// An iterable equivalent of [`Parser`], i.e: a parser that generates a sequence of outputs.
// TODO: Make sealed
pub trait IterParser<'a, I: Input + ?Sized, O, E: ParserExtra<'a, I> = extra::Default> {
    /// The state of the iterator during iteration.
    type IterState<M: Mode>
    where
        I: 'a;

    #[doc(hidden)]
    fn make_iter<M: Mode>(
        &self,
        inp: &mut InputRef<'a, '_, I, E>,
    ) -> PResult<Emit, Self::IterState<M>, E::Error>;
    #[doc(hidden)]
    fn next<M: Mode>(
        &self,
        inp: &mut InputRef<'a, '_, I, E>,
        state: &mut Self::IterState<M>,
    ) -> Option<PResult<M, O, E::Error>>;

    /// Collect this iterable parser into a [`Container`].
    ///
    /// This is commonly useful for collecting parsers that many values outputs into containers of various kinds:
    /// [`Vec`]s, [`String`]s, or even [`HashMap`]s. This method is analogous to [`Iterator::collect`].
    ///
    /// The output type of this parser is `C`, the type being collected into.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::{prelude::*, error::Simple};
    /// let word = any::<_, extra::Err<Simple<str>>>().filter(|c: &char| c.is_alphabetic()) // This parser produces an output of `char`
    ///     .repeated() // This parser is iterable (i.e: implements `IterParser`)
    ///     .collect::<String>(); // We collect the `char`s into a `String`
    ///
    /// assert_eq!(word.parse("hello").into_result(), Ok("hello".to_string()));
    /// ```
    fn collect<C: Container<O>>(self) -> Collect<Self, O, C>
    where
        Self: Sized,
    {
        Collect {
            parser: self,
            phantom: PhantomData,
        }
    }

    /// Collect this iterable parser into a [`usize`], outputting the number of elements that were parsed.
    ///
    /// This is sugar for [`.collect::<usize>()`](Self::collect).
    ///
    /// # Examples
    ///
    /// ```
    /// # use chumsky::zero_copy::prelude::*;
    ///
    /// // Counts how many chess squares are in the input.
    /// let squares = one_of::<_, _, extra::Err<Simple<str>>>('a'..='z').then(one_of('1'..='8')).padded().repeated().count();
    ///
    /// assert_eq!(squares.parse("a1 b2 c3").into_result(), Ok(3));
    /// assert_eq!(squares.parse("e5 e7 c6 c7 f6 d5 e6 d7 e4 c5 d6 c4 b6 f5").into_result(), Ok(14));
    /// assert_eq!(squares.parse("").into_result(), Ok(0));
    /// ```
    fn count(self) -> Collect<Self, O, usize>
    where
        Self: Sized,
    {
        self.collect()
    }

    /// Create an iterator over the outputs generated by an iterable parser.
    fn parse_iter(
        self,
        input: &'a I,
    ) -> ParseResult<ParserIter<'a, 'static, Self, I, O, E>, E::Error>
    where
        Self: IterParser<'a, I, O, E> + Sized,
        E::State: Default,
        E::Context: Default,
    {
        let mut inp = InputRef::new(input, Err(E::State::default()));
        ParseResult::new(
            Some(ParserIter {
                state: match self.make_iter::<Emit>(&mut inp) {
                    Ok(out) => out,
                    Err(e) => return ParseResult::new(None, vec![e.err]),
                },
                parser: self,
                inp,
                phantom: PhantomData,
            }),
            Vec::new(),
        )
    }

    /// Create an iterator over the outputs generated by an iterable parser with the given parser state.
    fn parse_iter_with_state<'parse>(
        self,
        input: &'a I,
        state: &'parse mut E::State,
    ) -> ParseResult<ParserIter<'a, 'parse, Self, I, O, E>, E::Error>
    where
        Self: IterParser<'a, I, O, E> + Sized,
        E::Context: Default,
    {
        let mut inp = InputRef::new(input, Ok(state));
        ParseResult::new(
            Some(ParserIter {
                state: match self.make_iter::<Emit>(&mut inp) {
                    Ok(out) => out,
                    Err(e) => return ParseResult::new(None, vec![e.err]),
                },
                parser: self,
                inp,
                phantom: PhantomData,
            }),
            Vec::new(),
        )
    }
}

/*
impl<'a, I, O, E, P> IterParser<'a, I, O::Item, E> for P
where
    I: Input + ?Sized + 'a,
    E: ParserExtra<'a, I>,
    P: Parser<'a, I, O, E>,
    O: IntoIterator,
{
    type IterState<M: Mode> = O::IntoIter;

    fn make_iter<M: Mode>(&self, inp: &mut InputRef<'a, '_, I, E>) -> PResult<M, Self::IterState<M>, E::Error> {
        Ok(M::map(self.go::<M>(inp)?, |xs: O| xs.into_iter()))
    }

    fn next(&self, inp: &mut InputRef<'a, '_, I, E>, state: &mut Self::IterState<Emit>) -> Option<PResult<Emit, O, E::Error>> {
        state.next().map(Ok)
    }
}
*/

/// An iterable equivalent of [`ConfigParser`], i.e: a parser that generates a sequence of outputs and
/// can be configured at runtime.
pub trait ConfigIterParser<'a, I: Input + ?Sized, O, E: ParserExtra<'a, I> = extra::Default>:
    IterParser<'a, I, O, E>
{
    /// Type used to configure the parser
    type Config: Default;

    #[doc(hidden)]
    fn next_cfg<M: Mode>(
        &self,
        inp: &mut InputRef<'a, '_, I, E>,
        state: &mut Self::IterState<M>,
        cfg: &Self::Config,
    ) -> Option<PResult<M, O, E::Error>>;

    /// A combinator that allows configuration of the parser from the current context
    fn configure<F>(self, cfg: F) -> IterConfigure<Self, F, O>
    where
        Self: Sized,
        F: Fn(Self::Config, &E::Context) -> Self::Config,
    {
        IterConfigure {
            parser: self,
            cfg,
            phantom: PhantomData,
        }
    }
}

/// See [`Parser::boxed`].
///
/// This type is a [`repr(transparent)`](https://doc.rust-lang.org/nomicon/other-reprs.html#reprtransparent) wrapper
/// around its inner value.
///
/// Due to current implementation details, the inner value is not, in fact, a [`Box`], but is an [`Rc`] to facilitate
/// efficient cloning. This is likely to change in the future. Unlike [`Box`], [`Rc`] has no size guarantees: although
/// it is *currently* the same size as a raw pointer.
// TODO: Don't use an Rc
pub struct Boxed<'a, I: Input + ?Sized, O, E: ParserExtra<'a, I>> {
    inner: Rc<dyn Parser<'a, I, O, E> + 'a>,
}

impl<'a, I: Input + ?Sized, O, E: ParserExtra<'a, I>> Clone for Boxed<'a, I, O, E> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, I, O, E> Parser<'a, I, O, E> for Boxed<'a, I, O, E>
where
    I: Input + ?Sized,
    E: ParserExtra<'a, I>,
{
    fn go<M: Mode>(&self, inp: &mut InputRef<'a, '_, I, E>) -> PResult<M, O, E::Error> {
        M::invoke(&*self.inner, inp)
    }

    go_extra!(O);
}

#[test]
fn zero_copy() {
    use self::input::WithContext;
    use self::prelude::*;

    // #[derive(Clone)]
    // enum TokenTest {
    //     Root,
    //     Branch(Box<Self>),
    // }

    // fn parser2() -> impl Parser<'static, str, TokenTest> {
    //     recursive(|token| {
    //         token
    //             .delimited_by(just('c'), just('c'))
    //             .map(Box::new)
    //             .map(TokenTest::Branch)
    //             .or(just('!').to(TokenTest::Root))
    //     })
    // }

    #[derive(PartialEq, Debug)]
    enum Token<'a> {
        Ident(&'a str),
        String(&'a str),
    }

    type FileId = u32;

    type Span = (FileId, SimpleSpan<usize>);

    fn parser<'a>() -> impl Parser<'a, WithContext<'a, FileId, str>, [(Span, Token<'a>); 6]> {
        let ident = any()
            .filter(|c: &char| c.is_alphanumeric())
            .repeated()
            .at_least(1)
            .map_slice(Token::Ident);

        let string = just('"')
            .then(any().filter(|c: &char| *c != '"').repeated())
            .then(just('"'))
            .map_slice(Token::String);

        ident
            .or(string)
            .map_with_span(|token, span| (span, token))
            .padded()
            .repeated_exactly()
            .collect()
    }

    assert_eq!(
        parser()
            .parse(&WithContext(42, r#"hello "world" these are "test" tokens"#))
            .into_result(),
        Ok([
            ((42, (0..5).into()), Token::Ident("hello")),
            ((42, (6..13).into()), Token::String("\"world\"")),
            ((42, (14..19).into()), Token::Ident("these")),
            ((42, (20..23).into()), Token::Ident("are")),
            ((42, (24..30).into()), Token::String("\"test\"")),
            ((42, (31..37).into()), Token::Ident("tokens")),
        ]),
    );
}

use crate::zero_copy::input::Emitter;
use combinator::MapSlice;

#[test]
fn zero_copy_repetition() {
    use self::prelude::*;

    fn parser<'a>() -> impl Parser<'a, str, Vec<u64>> {
        any()
            .filter(|c: &char| c.is_ascii_digit())
            .repeated()
            .at_least(1)
            .at_most(3)
            .map_slice(|b: &str| b.parse::<u64>().unwrap())
            .padded()
            .separated_by(just(',').padded())
            .allow_trailing()
            .collect()
            .delimited_by(just('['), just(']'))
    }

    assert_eq!(
        parser().parse("[122 , 23,43,    4, ]").into_result(),
        Ok(vec![122, 23, 43, 4]),
    );
    assert_eq!(
        parser().parse("[0, 3, 6, 900,120]").into_result(),
        Ok(vec![0, 3, 6, 900, 120]),
    );
    assert_eq!(
        parser().parse("[200,400,50  ,0,0, ]").into_result(),
        Ok(vec![200, 400, 50, 0, 0]),
    );

    assert!(parser().parse("[1234,123,12,1]").has_errors());
    assert!(parser().parse("[,0, 1, 456]").has_errors());
    assert!(parser().parse("[3, 4, 5, 67 89,]").has_errors());
}

#[test]
fn zero_copy_group() {
    use self::prelude::*;

    fn parser<'a>() -> impl Parser<'a, str, (&'a str, u64, char)> {
        group((
            any()
                .filter(|c: &char| c.is_ascii_alphabetic())
                .repeated()
                .at_least(1)
                .slice()
                .padded(),
            any()
                .filter(|c: &char| c.is_ascii_digit())
                .repeated()
                .at_least(1)
                .map_slice(|s: &str| s.parse::<u64>().unwrap())
                .padded(),
            any().filter(|c: &char| !c.is_whitespace()).padded(),
        ))
    }

    assert_eq!(
        parser().parse("abc 123 [").into_result(),
        Ok(("abc", 123, '[')),
    );
    assert_eq!(
        parser().parse("among3d").into_result(),
        Ok(("among", 3, 'd')),
    );
    assert_eq!(
        parser().parse("cba321,").into_result(),
        Ok(("cba", 321, ',')),
    );

    assert!(parser().parse("abc 123  ").has_errors());
    assert!(parser().parse("123abc ]").has_errors());
    assert!(parser().parse("and one &").has_errors());
}

#[cfg(feature = "regex")]
#[test]
fn regex_parser() {
    use self::prelude::*;
    use self::regex::*;

    fn parser<'a, C: Char>() -> impl Parser<'a, C::Slice, Vec<&'a C::Slice>> {
        regex("[a-zA-Z_][a-zA-Z0-9_]*")
            .padded()
            .repeated()
            .collect()
    }
    assert_eq!(
        parser::<char>()
            .parse("hello world this works")
            .into_result(),
        Ok(vec!["hello", "world", "this", "works"]),
    );

    assert_eq!(
        parser::<u8>()
            .parse(b"hello world this works" as &[_])
            .into_result(),
        Ok(vec![
            b"hello" as &[_],
            b"world" as &[_],
            b"this" as &[_],
            b"works" as &[_],
        ]),
    );
}

#[test]
fn unicode_str() {
    let input = "🄯🄚🹠🴎🄐🝋🰏🄂🬯🈦g🸵🍩🕔🈳2🬙🨞🅢🭳🎅h🵚🧿🏩🰬k🠡🀔🈆🝹🤟🉗🴟📵🰄🤿🝜🙘🹄5🠻🡉🱖🠓";
    let mut state = ();
    let mut input = InputRef::<_, extra::Default>::new(input, Ok(&mut state));

    while let (_, Some(c)) = input.next() {
        std::hint::black_box(c);
    }
}

#[test]
fn iter() {
    use self::prelude::*;

    fn parser<'a>() -> impl IterParser<'a, str, char> {
        any().repeated()
    }

    let mut chars = String::new();
    for c in parser().parse_iter("abcdefg").into_result().unwrap() {
        chars.push(c);
    }

    assert_eq!(&chars, "abcdefg");
}
