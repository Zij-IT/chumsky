//! Pratt parser for binary infix operators.
//!
//! Pratt parsing is an algorithm that allows efficient
//! parsing of binary infix operators.
//!
//! The [`binary_infix_operator()`] function creates a Pratt parser.
//! Its documentation contains an example of how it can be used.

use super::*;

use core::cmp;

/// Indicates which argument binds more strongly with a binary infix operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Associativity {
    /// The operator binds more strongly with the argument to the left.
    ///
    /// For example `a + b + c` is parsed as `(a + b) + c`.
    Left,

    /// The operator binds more strongly with the argument to the right.
    ///
    /// For example `a + b + c` is parsed as `a + (b + c)`.
    Right,
}

/// Indicates the binding strength of an operator to an argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Strength<T> {
    /// This is the strongly associated side of the operator.
    Strong(T),

    /// This is the weakly associated side of the operator.
    Weak(T),
}

impl<T> Strength<T> {
    /// Get the binding strength, ignoring associativity.
    pub fn strength(&self) -> &T {
        match self {
            Self::Strong(strength) => strength,
            Self::Weak(strength) => strength,
        }
    }
}

impl<T: Ord> Strength<T> {
    /// Compare two strengths.
    ///
    /// `None` is considered less strong than any `Some(Strength<T>)`,
    /// as it's used to indicate the lack of an operator
    /// to the left of the first expression and cannot bind.
    fn is_lt(&self, other: &Option<Self>) -> bool {
        match (self, other) {
            (x, Some(y)) => x < y,
            (_, None) => false,
        }
    }
}

impl<T: PartialOrd> PartialOrd for Strength<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.strength().partial_cmp(other.strength()) {
            Some(Ordering::Equal) => match (self, other) {
                (Self::Strong(_), Self::Weak(_)) => Some(cmp::Ordering::Greater),
                (Self::Weak(_), Self::Strong(_)) => Some(cmp::Ordering::Less),
                _ => Some(cmp::Ordering::Equal),
            },
            ord => ord,
        }
    }
}

impl<T: Ord> Ord for Strength<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Defines the parsing precedence of an operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InfixPrecedence<T> {
    strength: T,
    associativity: Associativity,
}

impl<T> InfixPrecedence<T> {
    /// Create a new precedence value.
    pub fn new(strength: T, associativity: Associativity) -> Self {
        Self {
            strength,
            associativity,
        }
    }
}

impl<T: Ord + Copy> InfixPrecedence<T> {
    /// Get the binding power of this operator with an argument on the left.
    fn strength_left(&self) -> Strength<T> {
        match self.associativity {
            Associativity::Left => Strength::Weak(self.strength),
            Associativity::Right => Strength::Strong(self.strength),
        }
    }

    /// Get the binding power of this operator with an argument on the right.
    fn strength_right(&self) -> Strength<T> {
        match self.associativity {
            Associativity::Left => Strength::Strong(self.strength),
            Associativity::Right => Strength::Weak(self.strength),
        }
    }
}

/// Enable Pratt parsing for a binary infix operator.
pub trait InfixOperator<Expr> {
    /// The type used to represent operator binding strength.
    ///
    /// Unless you have more than 256 operators,
    /// [`u8`] should be a good choice.
    type Strength: Copy + Ord;

    /// Get the parsing precedence of this operator.
    ///
    /// If an expression has an operator on both sides,
    /// the one with the greatest strength will
    /// be built first.
    ///
    /// For example, given `x + y * z` where `*` has a greater strength
    /// than `+` (as usual), the `y` will be combined with the `z` first.
    /// Next, the combination `(y * z)` will be combined with `x`,
    /// resulting in `(x + (y * z))`.
    ///
    /// If both sides have operators with the same strength,
    /// then the associativity will determine which side
    /// will be combined first.
    ///
    /// For example, given `x + y + z`;
    /// left-associativity will result in `((x + y) + z)`,
    /// while right-associativity will result in `(x + (y + z))`,
    fn precedence(&self) -> InfixPrecedence<Self::Strength>;

    /// Build an expression for this operator given two arguments.
    fn build_expression(self, left: Expr, right: Expr) -> Expr;
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
/// use chumsky::zero_copy::pratt::{binary_infix_operator, InfixOperator, InfixPrecedence, Associativity};
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
/// let atom = text::int::<_, (), (), _>(10)
///     .try_map(|int: &str, span|
///         int.parse()
///             .map_err(|_| ())
///     )
///     .map(Expr::Literal);
/// let operator = choice((
///     just('+').to(Operator::Add),
///     just('-').to(Operator::Sub),
///     just('*').to(Operator::Mul),
///     just('/').to(Operator::Div),
/// ));
///
/// let expr = binary_infix_operator(atom, operator.padded_by(just(' ')));
/// let expr_str = expr.map(|expr: Expr| expr.to_string()).then_ignore(end());
/// assert_eq!(expr_str.parse("1 + 2"), (Some("(1 + 2)".to_string()), vec![]));
/// // `*` binds more strongly than `+`
/// assert_eq!(expr_str.parse("1 * 2 + 3"), (Some("((1 * 2) + 3)".to_string()), vec![]));
/// assert_eq!(expr_str.parse("1 + 2 * 3"), (Some("(1 + (2 * 3))".to_string()), vec![]));
/// // `+` is left-associative
/// assert_eq!(expr_str.parse("1 + 2 + 3"), (Some("((1 + 2) + 3)".to_string()), vec![]));
/// // `*` is right-associative (in this example)
/// assert_eq!(expr_str.parse("1 * 2 * 3"), (Some("(1 * (2 * 3))".to_string()), vec![]));
/// ```
pub const fn binary_infix_operator<I: ?Sized, E, S, Op, Expr, AtomParser, OpParser>(
    atom_parser: AtomParser,
    operator_parser: OpParser,
) -> BinaryInfixOperator<I, E, S, Op, Expr, AtomParser, OpParser> {
    BinaryInfixOperator {
        atom_parser,
        operator_parser,
        phantom: PhantomData,
    }
}

/// See [`binary_infix_operator()`].
pub struct BinaryInfixOperator<I: ?Sized, E, S, Op, Expr, AtomParser, OpParser> {
    atom_parser: AtomParser,
    operator_parser: OpParser,
    phantom: PhantomData<(E, S, Op, Expr, I)>,
}

impl<I, E, S, Op, Expr, AtomParser: Copy, OpParser: Copy> Copy
    for BinaryInfixOperator<I, E, S, Op, Expr, AtomParser, OpParser>
{
}

impl<I: ?Sized, E, S, Op, Expr, AtomParser: Clone, OpParser: Clone> Clone
    for BinaryInfixOperator<I, E, S, Op, Expr, AtomParser, OpParser>
{
    fn clone(&self) -> Self {
        Self {
            atom_parser: self.atom_parser.clone(),
            operator_parser: self.operator_parser.clone(),
            phantom: PhantomData,
        }
    }
}

impl<'a, I, E, S, Op, Expr, AtomParser, OpParser>
    BinaryInfixOperator<I, E, S, Op, Expr, AtomParser, OpParser>
where
    I: Input + ?Sized,
    E: Error<I>,
    S: 'a,
    Op: InfixOperator<Expr>,
    AtomParser: Parser<'a, I, E, S, Output = Expr>,
    OpParser: Parser<'a, I, E, S, Output = Op>,
{
    fn pratt_parse(
        &self,
        inp: &mut InputRef<'a, '_, I, E, S>,
        min_strength: Option<Strength<Op::Strength>>,
    ) -> PResult<Emit, Expr, E> {
        let mut left_expr: Expr = self.atom_parser.go::<Emit>(inp)?;

        loop {
            let offset = inp.save();
            let (op, prec) = match self.operator_parser.go::<Emit>(inp) {
                Ok(op) => {
                    let precedence = op.precedence();
                    if precedence.strength_left().is_lt(&min_strength) {
                        inp.rewind(offset);
                        return Ok(Emit::bind(|| left_expr));
                    }
                    (op, precedence)
                }
                Err(_) => return Ok(Emit::bind(|| left_expr)),
            };

            match self.pratt_parse(inp, Some(prec.strength_right())) {
                Ok(right_expr) => {
                    left_expr = op.build_expression(left_expr, right_expr);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

impl<'a, I, E, S, Op, Expr, AtomParser, OpParser> Parser<'a, I, E, S>
    for BinaryInfixOperator<I, E, S, Op, Expr, AtomParser, OpParser>
where
    I: Input + ?Sized,
    E: Error<I>,
    S: 'a,
    Op: InfixOperator<Expr>,
    AtomParser: Parser<'a, I, E, S, Output = Expr>,
    OpParser: Parser<'a, I, E, S, Output = Op>,
{
    type Output = Expr;

    fn go<M: Mode>(&self, inp: &mut InputRef<'a, '_, I, E, S>) -> PResult<M, Self::Output, E> {
        let val = self.pratt_parse(inp, None)?;
        Ok(M::bind(|| val))
    }

    go_extra!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    enum Operator {
        Add,
        Sub,
        Mul,
        Div,
    }

    enum Expr {
        Literal(i64),
        Add(Box<Expr>, Box<Expr>),
        Sub(Box<Expr>, Box<Expr>),
        Mul(Box<Expr>, Box<Expr>),
        Div(Box<Expr>, Box<Expr>),
    }

    impl std::fmt::Display for Expr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Literal(literal) => write!(f, "{literal}"),
                Self::Add(left, right) => write!(f, "({left} + {right})"),
                Self::Sub(left, right) => write!(f, "({left} - {right})"),
                Self::Mul(left, right) => write!(f, "({left} * {right})"),
                Self::Div(left, right) => write!(f, "({left} / {right})"),
            }
        }
    }

    impl InfixOperator<Expr> for Operator {
        type Strength = u8;

        fn precedence(&self) -> InfixPrecedence<Self::Strength> {
            match self {
                Self::Add => InfixPrecedence::new(0, Associativity::Left),
                Self::Sub => InfixPrecedence::new(0, Associativity::Left),
                Self::Mul => InfixPrecedence::new(1, Associativity::Right),
                Self::Div => InfixPrecedence::new(1, Associativity::Right),
            }
        }

        fn build_expression(self, left: Expr, right: Expr) -> Expr {
            let (left, right) = (Box::new(left), Box::new(right));
            match self {
                Self::Add => Expr::Add(left, right),
                Self::Sub => Expr::Sub(left, right),
                Self::Mul => Expr::Mul(left, right),
                Self::Div => Expr::Div(left, right),
            }
        }
    }

    fn parser<'a>() -> impl Parser<'a, str, (), (), Output = String> {
        let atom = text::int(10)
            .try_map(|int: &str, _span| int.parse().map_err(|_| ()))
            .map(Expr::Literal);

        let operator = super::prelude::choice((
            super::prelude::just('+').to(Operator::Add),
            super::prelude::just('-').to(Operator::Sub),
            super::prelude::just('*').to(Operator::Mul),
            super::prelude::just('/').to(Operator::Div),
        ));

        binary_infix_operator::<str, (), (), Operator, Expr, _, _>(atom, operator)
            .map(|expr: Expr| expr.to_string())
    }

    fn complete_parser<'a>() -> impl Parser<'a, str, (), (), Output = String> {
        parser().then_ignore(super::prelude::end())
    }

    fn parse(input: &str) -> (Option<String>, Vec<()>) {
        complete_parser().parse(input)
    }

    fn parse_partial(input: &str) -> (Option<String>, Vec<()>) {
        parser().parse(input)
    }

    #[test]
    fn missing_first_expression() {
        assert_eq!(parse(""), (None, vec![()]),);
    }

    #[test]
    fn missing_later_expression() {
        assert_eq!(parse("1+"), (None, vec![()]),);
    }

    #[test]
    fn invalid_first_expression() {
        assert_eq!(parse("?"), (None, vec![()]),);
    }

    #[test]
    fn invalid_later_expression() {
        assert_eq!(parse("1+?"), (None, vec![()]),);
    }

    #[test]
    fn invalid_operator() {
        assert_eq!(parse("1?"), (None, vec![()]),);
    }

    #[test]
    fn invalid_operator_incomplete() {
        assert_eq!(parse_partial("1?"), (Some("1".to_string()), vec![]),);
    }

    #[test]
    fn complex_nesting() {
        assert_eq!(
            parse_partial("1+2*3/4*5-6*7+8-9+10"),
            (
                Some("(((((1 + (2 * (3 / (4 * 5)))) - (6 * 7)) + 8) - 9) + 10)".to_string()),
                vec![]
            ),
        );
    }
}
