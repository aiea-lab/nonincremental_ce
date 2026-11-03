"""Logical formulas of concepts. Adapted from the formula module of
https://github.com/jayelm/compexp.
"""


class F:
    """Base class of all formulas."""


def compute_hash_value(f: F) -> tuple:
    """Returns a representation of the formula that does not depend on the
    order of the operands of chains of the same operator."""
    if isinstance(f, Leaf):
        return ("LEAF", f.val)
    if isinstance(f, Not):
        return ("NOT", compute_hash_value(f.val))
    if isinstance(f, (Or, And)):
        # Flatten the chain of the same operator
        operands = []
        nodes = [f]
        while nodes:
            node = nodes.pop()
            if isinstance(node, type(f)):
                nodes.extend([node.left, node.right])
            else:
                operands.append(compute_hash_value(node))
        return (f.op, tuple(sorted(operands)))
    raise TypeError(f"Unsupported formula type: {type(f)}")


class Leaf(F):
    """An atomic concept."""

    def __init__(self, val: int):
        self.val = val

    def __str__(self) -> str:
        return str(self.val)

    def __repr__(self) -> str:
        return f"Leaf({str(self)})"

    def __len__(self) -> int:
        return 1

    def __hash__(self) -> int:
        return hash(compute_hash_value(self))

    def __eq__(self, other: object) -> bool:
        """Two leaves are equal if they have the same concept."""
        if not isinstance(other, Leaf):
            return False
        return self.val == other.val

    def __lt__(self, other: F) -> bool:
        """Orders the leaves before the other formulas."""
        if isinstance(other, Leaf):
            return self.val < other.val
        return True

    def get_vals(self) -> list:
        """Returns the concept of the leaf."""
        return [self.val]

    def get_leaves(self) -> list:
        """Returns the leaf itself."""
        return [self]

    def get_ops(self) -> list:
        """Returns the operators of the formula."""
        return []


class UnaryNode(F):
    """A node with a single operand."""

    op = None

    def __init__(self, val: F):
        self.val = val

    def __str__(self) -> str:
        return f"({self.op} {self.val})"

    def __repr__(self) -> str:
        return f"{self.op}({self.val})"

    def __len__(self) -> int:
        return len(self.val)

    def get_vals(self) -> list:
        """Returns the concept of the operand."""
        return self.val.get_vals()


class BinaryNode(F):
    """A node with two operands."""

    op = None

    def __init__(self, left: F, right: F):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"

    def __repr__(self) -> str:
        return f"{self.op}({self.left}, {self.right})"

    def __len__(self) -> int:
        return len(self.left) + len(self.right)

    def get_vals(self) -> list:
        """Returns the concepts of the formula, starting from the right subtree."""
        return self.right.get_vals() + self.left.get_vals()

    def get_leaves(self) -> list:
        """Returns the leaves of the formula, starting from the right subtree."""
        return self.right.get_leaves() + self.left.get_leaves()

    def get_ops(self) -> list:
        """Returns the operators of the formula."""
        ops = [self.op]
        for child in (self.left, self.right):
            if isinstance(child, (BinaryNode, UnaryNode)):
                ops.extend(child.get_ops())
        return ops


class Not(UnaryNode):
    """The NOT operator."""

    op = "NOT"

    def __hash__(self) -> int:
        return hash(compute_hash_value(self))

    def __eq__(self, other: object) -> bool:
        """Two NOT nodes are equal if they negate the same concept."""
        if not isinstance(other, Not):
            return False
        return self.val == other.val

    def __lt__(self, other: F) -> bool:
        """Orders the NOT nodes after the other formulas."""
        if isinstance(other, Not):
            return self.val < other.val
        return False

    def get_ops(self) -> list:
        """Returns the operator NOT."""
        assert isinstance(self.val, Leaf)
        return [self.op]

    def get_leaves(self) -> list:
        """Returns the negated concept as a single leaf."""
        return [self]


class Or(BinaryNode):
    """The OR operator."""

    op = "OR"

    def __hash__(self) -> int:
        return hash(compute_hash_value(self))

    def __eq__(self, other: object) -> bool:
        """Two OR formulas are equal up to the order of their operands."""
        if not isinstance(other, Or):
            return False
        # Chains of OR are equal if they contain the same concepts
        if len(set(self.get_ops())) == 1 and len(set(other.get_ops())) == 1:
            return sorted(self.get_vals()) == sorted(other.get_vals())
        # Otherwise, the two operands can be swapped
        return (self.left == other.left and self.right == other.right) or (
            self.left == other.right and self.right == other.left
        )

    def __lt__(self, other: F) -> bool:
        """Orders the OR formulas after the leaves and before the AND formulas."""
        if isinstance(other, Or):
            return self.left < other.left or (
                self.left == other.left and self.right < other.right
            )
        if isinstance(other, Leaf):
            return False
        return True


def is_full_not(formula: F) -> bool:
    """Returns whether the formula is a chain of AND NOT."""
    ops = formula.get_ops()
    if "OR" in ops:
        return False
    return "AND" in ops and "NOT" in ops and ops.count("AND") == ops.count("NOT")


class And(BinaryNode):
    """The AND operator."""

    op = "AND"

    def __init__(self, left: F, right: F):
        assert not isinstance(left, Not), "NOT terms must appear on the right side of AND"
        super().__init__(left, right)

    def __hash__(self) -> int:
        return hash(compute_hash_value(self))

    def __eq__(self, other: object) -> bool:
        """Two AND formulas are equal up to the order of their operands."""
        if not isinstance(other, And):
            return False
        # Chains of AND are equal if they contain the same concepts
        if len(set(self.get_ops())) == 1 and len(set(other.get_ops())) == 1:
            return sorted(self.get_vals()) == sorted(other.get_vals())

        self_is_not = isinstance(self.right, Not)
        other_is_not = isinstance(other.right, Not)
        if self_is_not and other_is_not:
            # Chains of AND NOT are equal if they share the first concept and
            # contain the same concepts
            if is_full_not(self) and is_full_not(other):
                self_vals, other_vals = self.get_vals(), other.get_vals()
                if self_vals[-1] != other_vals[-1]:
                    return False
                return sorted(self_vals) == sorted(other_vals)
            return self.left == other.left and self.right.val == other.right.val
        if not self_is_not and not other_is_not:
            # The two operands can be swapped
            return (self.left == other.left and self.right == other.right) or (
                self.left == other.right and self.right == other.left
            )
        # One is AND, the other is AND NOT
        return False

    def __lt__(self, other: F) -> bool:
        """Orders the AND formulas after the OR formulas and before the NOT
        nodes."""
        if isinstance(other, And):
            return self.left < other.left or (
                self.left == other.left and self.right < other.right
            )
        if isinstance(other, (Leaf, Or)):
            return False
        return True


def get_formula_str(f: F, concept_names: list) -> str:
    """Returns the formula as a string, replacing indices with concept names."""
    if isinstance(f, And):
        return f"({get_formula_str(f.left, concept_names)} AND {get_formula_str(f.right, concept_names)})"
    if isinstance(f, Or):
        return f"({get_formula_str(f.left, concept_names)} OR {get_formula_str(f.right, concept_names)})"
    if isinstance(f, Not):
        return f"NOT {get_formula_str(f.val, concept_names)}"
    if isinstance(f, Leaf):
        return concept_names[f.val]


def from_optimal_formula(optimal_formula):
    """Converts a formula of the compositional_explanations package."""
    from compositional_explanations import formula as optimal_F

    if isinstance(optimal_formula, optimal_F.Leaf):
        return Leaf(optimal_formula.val)
    if isinstance(optimal_formula, optimal_F.Not):
        return Not(from_optimal_formula(optimal_formula.val))
    if isinstance(optimal_formula, optimal_F.Or):
        return Or(from_optimal_formula(optimal_formula.left), from_optimal_formula(optimal_formula.right))
    if isinstance(optimal_formula, optimal_F.And):
        return And(from_optimal_formula(optimal_formula.left), from_optimal_formula(optimal_formula.right))
    raise ValueError(f"Unknown formula type: {type(optimal_formula)}")
