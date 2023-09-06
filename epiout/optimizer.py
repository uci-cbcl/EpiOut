import collections
import tensorflow as tf


_LineSearchStep = collections.namedtuple(
    'Step', 'x, args, loss, grad, direction')
_LineSearchResult = collections.namedtuple('Result', 'position, history')


class BackwardLineSearchGD:

    def __init__(self, loss_fn, max_iterations=10, tol=1e-5, decay_rate=0.5,
                 c1=10**-4, c2=0.9, boundary=None, parallel_iterations=10):
        '''
        Backward line search algorithm with gradient decent.

        Args:
          loss_fn: Loss function takes x (variable to be optmized) and
            returns loss.
          max_iterations: Maximum number of line search iterations.
          tol: Tolerance to stop optimziation.
          decay_rate: Decay rate of line-search.
          c1: Constant for armijo rule.
          c2: Curvature condition.
          boundary: If bounded optimization upper and lower bounds.
          parallel_iterations: Number of paralel iterations.
        '''
        self.loss_fn = loss_fn
        self.max_iterations = max_iterations
        self.tol = tol
        self.decay_rate = decay_rate
        self.c1 = c1
        self.c2 = c2
        self.boundary = boundary
        self.parallel_iterations = parallel_iterations

    def _update(self, x, grad, lr):
        '''
        Update variable based on gradients and learning rate
          and clip if values outside of boundries.

        Args:
          x: Variable to update
          grad: Gradients
          lr: Learning rate
        '''
        x_new = x - grad * lr

        if self.boundary:
            lower, upper = self.boundary
            x_new = tf.clip_by_value(x_new, lower, upper)

        return x_new

    def _wilfe_condition(self, lr, step):
        '''
        Wilfe condition

        https://en.wikipedia.org/wiki/Wolfe_conditions

        Args:
          lr: Initial learning rate.
          step: _LineSearchStep object 
        '''
        x_new = self._update(step.x, step.grad, lr)

        loss_new, grad_new = self._loss_grad(x_new, step.args)

        d_grad = step.direction * step.grad
        d_grad_new = step.direction * grad_new

        # sufficient loss decrease
        armijo_rule = (loss_new > step.loss - self.c1 * lr * d_grad)
        # sufficient  slope reduce
        curvature_condition = (-d_grad_new > self.c2 * d_grad)

        return armijo_rule | curvature_condition

    def _lr_decay(self, cond, lr, step):
        '''
        Decay learning rate for the conditions.
        '''
        cond = self._wilfe_condition(lr, step)
        lr = tf.where(cond, lr * self.decay_rate, lr)

        return cond, lr, step

    def _line_search_cond(self, cond, lr, step):
        '''
        Stop condition of line search. If all conditions 
          are False linear search stops.
        '''
        step_size = tf.math.abs(step.grad * lr)

        return tf.reduce_any(cond & (step_size > self.tol))

    def _line_search(self, inputs):
        '''
        Backward line search.

        Args:
          x: Variable to optimize (input of the loss function).
          loss: Loss values 
          grad: Gradients vector
          direction: Direction of line search (same with gradients
            if univariate function)
        '''
        x, args, loss, grad, direction = inputs

        lr_init = tf.ones_like(loss)
        cond_init = tf.constant(True, shape=loss.shape)
        step = _LineSearchStep(x, args, loss, grad, direction)

        return tf.while_loop(
            self._line_search_cond,
            self._lr_decay,
            [cond_init, lr_init, step]
        )[1]

    def line_search(self, x, args, loss, grad, direction):
        return tf.vectorized_map(self._line_search, (x, args, loss, grad, direction))

    def _loss_grad(self, x, args):
        '''
        Calcualtes loss and grad given input `x`.

        Args:
          x: Variable to optimize (input of the loss function).
        '''
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = self.loss_fn(x, *args)
        grad = tape.gradient(loss, x)

        return loss, grad

    @tf.function
    def _fit(self, x, args=()):
        '''
        One iteration of gradient decent with line search.

        Args:
          x: Variable to optimize (input of the loss function).
        '''
        loss, grad = self._loss_grad(x, args)

        direction = grad * 1.  # inv_hession = 1
        # for univariable function direction=grad

        lr = self.line_search(x, args, loss, grad, direction)

        x_new = self._update(x, grad, lr)
        loss_new = self.loss_fn(x_new, *args)

        # needed due to boundries
        x_new = tf.where(loss_new < loss, x_new, x)
        loss_new = tf.where(loss_new < loss, loss_new, loss)

        return x_new, loss_new

    def fit(self, initial_position, args=(), history=False):
        '''
        Optimize in variable in paralel

        Args:
          initial_position: Initial values of variable.
          history: Store loss history in the output.
        '''
        x = initial_position

        hist_loss = list()

        for _ in range(self.max_iterations):

            x_new, loss_new = self._fit(x, args)

            if history:
                hist_loss.append(loss_new)

            if tf.reduce_all(tf.abs(x - x_new) <= self.tol):
                break

            x = x_new

        return _LineSearchResult(
            position=x_new, history=tf.stack(hist_loss))


def backward_linesearch_gd(
        loss_fn, initial_position, args=(),
        max_iterations=10, tol=1e-5, decay_rate=0.5,
        c1=10**-4, c2=0.9, boundary=None, parallel_iterations=10):

    return BackwardLineSearchGD(
        loss_fn, max_iterations=max_iterations, tol=tol,
        decay_rate=decay_rate, c1=c1, c2=c2, boundary=boundary,
        parallel_iterations=parallel_iterations
    ).fit(initial_position, args).position
