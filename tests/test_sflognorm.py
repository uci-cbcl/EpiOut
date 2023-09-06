# import tensorflow as tf
# from aeoutlier.sflognorm import SFLogNorm


# def test_sflognorm():

#     X = torch.Tensor([
#         [100, 50, 10],
#         [10, 20, 30],
#         [1000, 500, 300],
#         [4, 2, 1],
#     ])

#     norm = SFLogNorm()
#     X_ = norm.fit_transform(X)

#     # TOTEST:
#     # torch.testing.assert_close(
#     #     norm.size_factor_,
#     #     torch.Tensor([[0], [1114],  [572],  [341]])
#     # )

#     torch.testing.assert_close(
#         X_.sum(axis=1),
#         torch.zeros_like(X_.sum(axis=1))
#     )

#     torch.testing.assert_close(
#         norm.inverse_transform(X_),
#         X
#     )
