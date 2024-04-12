from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict, Set

from typing_extensions import Protocol, Sequence

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals = list(vals)
    vals[arg] = vals[arg] + epsilon / 2
    diff_forward = f(*vals)
    vals[arg] = vals[arg] - epsilon
    diff_backward = f(*vals)
    return (diff_forward - diff_backward) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass
    
def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    def _reverse_dfs(variable: Variable, marked: Set[int], onpath: Set[int], res: List[Variable]) -> None:
        marked.add(variable.unique_id)
        onpath.add(variable.unique_id)
        # filt out constant variable
        if variable.is_constant():
            onpath.remove(variable.unique_id)
            return
        
        for child in variable.parents:
            if not child.unique_id in marked:
                _reverse_dfs(child, marked, onpath, res)
            if child.unique_id in onpath:
                raise RuntimeError("A cycle is detected in your computing graph!")
            
        res.insert(0, variable)
        onpath.remove(variable.unique_id)
    
    res: List[Variable] = []
    marked: Set[int] = set()
    onpath: Set[int] = set()
    _reverse_dfs(variable, marked, onpath, res)
    return res
    
    

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    var2deriv: Dict[int, Any] = {}
    topo_res: List[Variable] = topological_sort(variable)
    for cur_var in topo_res:
        assert not cur_var.is_constant()
        if cur_var.unique_id in var2deriv:
            deriv = var2deriv[cur_var.unique_id]
        if cur_var.is_leaf():
            cur_var.accumulate_derivative(deriv)
            continue
        var_derives = cur_var.chain_rule(deriv)
        for var, var_deriv in var_derives:
            if var.unique_id in var2deriv:
                var2deriv[var.unique_id] += var_deriv
            else:
                var2deriv[var.unique_id] = var_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
