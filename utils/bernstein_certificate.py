"""Global common-Bernstein bound for the unchanged cardinal cubic physical FFD.

See docs/goal05r/bernstein_certificate.rst for the convex-coefficient proof.
The safety limit is not defined here; this function only evaluates a bound.
"""
import torch


def bernstein_frobenius_bound(grid,spacing):
    """Common degree3 Bernstein coefficients of the complete Jacobian matrix.

    B converts the four cubic cardinal basis functions into Bernstein3.
    A converts the three positive quadratic derivative-difference weights into
    Bernstein3 by degree elevation. Rows of both matrices are nonnegative and
    sum to one. Every matrix column then has the SAME tensor-product Bernstein
    basis, so J(x) is a convex combination of complete3x3 coefficient matrices.
    max_coefficient ||J||_F bounds ||J(x)||_F and hence its operator norm globally.
    """
    leading = grid.shape[:-4]
    flat = grid.reshape(-1,*grid.shape[-4:])
    cubic = grid.new_tensor([[1/6,4/6,1/6,0.], [0.,4/6,2/6,0.],
                            [0.,2/6,4/6,0.], [0.,1/6,4/6,1/6]])
    quadratic_elevated = grid.new_tensor([[.5,.5,0.], [1/6,5/6,0.],
                                         [0.,5/6,1/6], [0.,.5,.5]])
    squared_frobenius = None
    for axis in range(3):
        delta = flat.diff(dim=1+axis)/spacing[axis]
        kernel = [3 if a == axis else 4 for a in range(3)]
        windows = delta.unfold(1,kernel[0],1).unfold(2,kernel[1],1).unfold(3,kernel[2],1)
        matrices = [quadratic_elevated if a == axis else cubic for a in range(3)]
        coefficients = torch.einsum('...oijk,ai->...oajk',windows,matrices[0])
        coefficients = torch.einsum('...oajk,bj->...oabk',coefficients,matrices[1])
        coefficients = torch.einsum('...oabk,ck->...oabc',coefficients,matrices[2])
        column_square = coefficients.square().sum(-4)
        squared_frobenius = column_square if squared_frobenius is None else squared_frobenius+column_square
    return squared_frobenius.flatten(1).amax(-1).sqrt().reshape(leading)

