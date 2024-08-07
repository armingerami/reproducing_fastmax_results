# This is the JAX version. The flags (mask, normalize, create_attn_matrix) should not take bool (True, False) values
# If create_attn_matrix == 1, to apply the mask, the mask should be given as a JAX N by N matrix
import jax

import jax.numpy as jnp
from jax import vmap

def fastmax(q, k, v, mask=None, denum_term=8, normalize = 0, create_attn_matrix = 0):
  """
  Input: query, key, and value matrices (b, h, n, d)
      b: batch size
      h: number of heads
      n: number of tokens
      d: dimension per attention head (d = d_model / h)
      mask: indicating whether to apply causal masking; mask = None means no mask, and mask = 1 means apply mask
        (Note: value of mask cannot be boolean for JAX. Just set mask to an int like 1 to apply the causal mask)
      denum_term: Hyperparameter to control the standard deviation of <q, k>; stdev(<q, k>) = 1/denum_term
        Stdev of <q, k> is important in general with attention, but even more so when using a taylor
        expansion to approximate an exponential because the error increases with the stdev of <q, k>.
        In normal attention, stdev equates to the "temperature" of the softmax function, and with a
        taylor approximation, higher temperature also means we drift further from the true softmax.
        For positive inputs, this drifting error actually lowers the temperature, and for negative inputs
        it raises the temperature.
  Output: The result of Attention matrix * Value (b, h, n, d)
  """
  if normalize == 1:
      denum_term = 1
      q -= jnp.mean(q)
      k -= jnp.mean(k)
      qn = jnp.linalg.norm(q, axis = 3)
      kn = jnp.linalg.norm(k, axis = 3)
      q /= jnp.linalg.norm(qn, axis = 2, ord = float('inf')).reshape((q.shape[0],q.shape[1],1,1))
      k /= jnp.linalg.norm(kn, axis = 2, ord = float('inf')).reshape((q.shape[0],q.shape[1],1,1))
  else:
      denum_term = denum_term*jnp.sqrt(q.shape[3])
  denum_term2 = 2*denum_term*denum_term

  # Prepare the quadratic terms with respect to k and q:
  q2 = jnp.matmul(q.reshape(k.shape[0],k.shape[1],k.shape[2],k.shape[3],1),q.reshape(k.shape[0],k.shape[1],k.shape[2],1,k.shape[3])).reshape(k.shape[0],k.shape[1],k.shape[2],k.shape[3]*k.shape[3])
  k2 = jnp.matmul(k.reshape(k.shape[0],k.shape[1],k.shape[2],k.shape[3],1),k.reshape(k.shape[0],k.shape[1],k.shape[2],1,k.shape[3])).reshape(k.shape[0],k.shape[1],k.shape[2],k.shape[3]*k.shape[3])

  if create_attn_matrix == 0:
    if mask is None:
        first_term = jnp.sum(v,-2)

        second_term = jnp.matmul(k.swapaxes(-2,-1),v)
        second_term /= denum_term

        third_term = jnp.matmul(k2.swapaxes(-2,-1),v)
        third_term /= denum_term2

        div1 = (jnp.ones([k.shape[0],k.shape[1],1])*k.shape[2]).reshape((k.shape[0],k.shape[1],1,1))
        div2 = jnp.sum(k,-2).reshape((k.shape[0],k.shape[1],k.shape[3],1))
        div3 = jnp.sum(k2,-2).reshape((k.shape[0],k.shape[1],k.shape[3]*k.shape[3],1))

        ans2 = jnp.matmul(q,second_term)
        ans3 = jnp.matmul(q2,third_term)
        div2 = jnp.matmul(q,div2)/ denum_term
        div3 = jnp.matmul(q2,div3)/ denum_term2

        ans = ans2+ans3
        ans = jnp.add(ans.swapaxes(-2,-1).T ,first_term.T).T.swapaxes(-2,-1)
        div = div2+div3
        div = jnp.add(div.swapaxes(-2,-1).T ,div1.T).T.swapaxes(-2,-1)
        ans = ans/div
    else:
        first = jnp.cumsum(v,2) # (b, h, n, d)
        second = jnp.einsum("bhij,bhijk->bhik",q, jnp.cumsum(jnp.einsum("bhij,bhik->bhijk",k,v),2))/denum_term # (b, h, n, d)
        third = jnp.einsum("bhij,bhijk->bhik",q2, jnp.cumsum(jnp.einsum("bhij,bhik->bhijk",k2,v),2))/denum_term2 # (b, h, n, d)

        kcs = jnp.cumsum(k,-2) # (b, h, n, d)
        k2cs = jnp.cumsum(k2,-2) # (b, h, n, d^2)
        div1 = jnp.cumsum(jnp.ones([q.shape[0],q.shape[1],q.shape[2]]),2) # (b, h, 1)
        div2 = jnp.einsum("bhij,bhij -> bhi",q,kcs)/denum_term # (b, h, n)
        div3 = jnp.einsum("bhij,bhij -> bhi",q2,k2cs)/denum_term2 # (b, h, n)
        div = div1 + div2 + div3 # (b, h, n, 1)
        div = jnp.expand_dims(div,-1) # (b, h, n, 1)

        ans = first + second + third # (b, h, n, d)
        ans /= div # (b, h, n, d)

    return ans
  
  else:
    attn = 1 + jnp.matmul(q, jnp.swapaxes(k, -2, -1))/denum_term + jnp.matmul(q2, jnp.swapaxes(k2, -2, -1))/denum_term2
    if mask is not None:
        attn = jnp.where(mask == 0, 0, attn)
    attn /= (jnp.sum(attn, axis=3)).reshape((attn.shape[0],attn.shape[1],attn.shape[2],1))
    ans = jnp.matmul(attn,v)
    return ans