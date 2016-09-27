
# logsumexp.py
# Numerically stable logsumexp (logadd) reduction along a tensor axis.

# Authors: Pascal Vincent and Alexandre de Brebisson (2016)
# License: same license as theano

import sys

import numpy as np
# import scipy

from six import integer_types
from theano.gof import Apply, Op, OpenMPOp
from theano import tensor as T
from theano import scalar 
from theano import config, shared, printing, pp
from theano import function, function_dump 
from theano.tensor import basic as tensor, subtensor, opt, elemwise, TensorType
from theano.tensor.opt import register_canonicalize, register_stabilize, register_specialize, gof
from theano.tensor.nnet import softplus, softmax_op

def softmax_npy(x, axis=-1):
    # encountered a numpy bug in some cases when using
    # numpy.amax with keepdims=True so using reshape instead
    keepdim_shape = list(x.shape)
    keepdim_shape[axis]=1
    x_max = x.max(axis=axis).reshape(keepdim_shape)
    x_exp = np.exp(x-x_max)
    result = x_exp/x_exp.sum(axis=axis).reshape(keepdim_shape)
    return result

def logsumexp_npy(x, axis=-1, keepdims=False):
    # encountered a numpy bug in some cases when using
    # numpy.amax with keepdims=True so using reshape instead
    keepdim_shape = list(x.shape)
    keepdim_shape[axis]=1
    x_max = x.max(axis=axis)
    result = x_max + np.log(np.sum( np.exp(x-x_max.reshape(keepdim_shape)), axis=axis))
    if keepdims: 
        result = result.reshape(keepdim_shape)
    # print >> sys.stderr, "logsumexp_npy x.dtype:", x.dtype, "x.shape:",x.shape," result.shape: ", result.shape
    return result


def softmax_with_axis(x, axis=-1):
    """This function computes/returns a softmax done on a tensor (ndarray or theano tensor variable)
    along a specified axis. It is a generalization of theano's arbitrarily restrictive softmax,
    which only handles matrix (2d tensor) with the softmax done along rows.
    """

    if axis is None:
        raise ValueError("softmax_with_axis only support a constant integer axis")

    if isinstance(x, np.ndarray):
        return softmax_npy(x, axis)

    else: # return theano graph variable
        if axis<0:
            axis = x.ndim+axis
        if axis>=x.ndim or axis<0:
            raise ValueError("specified axis value out of tensor ndim")

        if x.ndim<=0:
            raise ValueError("softmax_with_axis does not support scalar input")
        elif x.ndim==1:
            return softmax_op(T.shape_padleft(x,n_ones=1))[0,:]

        elif x.ndim==2: # simple softmax case
            if axis==1 or axis==-1:
                return softmax_op(x)
            else:
                return softmax_op(x.swapaxes(axis,1)).swapaxes(1,axis)

        # general case for dimension>2
        axis_size = x.shape[axis]
        x_swapped = x.swapaxes(axis,x.ndim-1)
        x_flat = x_swapped.flatten().reshape((x_swapped.size//axis_size, axis_size))
        sm = softmax_op(x_flat).reshape(x_swapped.shape)
        sm_reswapped = sm.swapaxes(axis,x.ndim-1)
        return sm_reswapped



class LogSumExp(Op):
    """
    This theano op computes a log(sum(exp(X),axis)) i.e. a log-sum-exp (logadd) reduction
    along a specified axis in a numerically stable fashion.
    It also returns a numerically stable gradient as a Softmax theano op.

    Specifically: for the forward computation, we provide a stable numpy version, as well as a theano optimisation that
      will substitute this op by  numerically stable expression (subtracting the max along the axis) that theano
      will know how to compile to efficient cpu or gpu code.

      As for the gradient, it consists of a 2d (i.e. matrix) softmax with row-reduction (because that's the only softmax that theano currently handles)
      with eventual pre and post dimshuffling/resizing if needed to be applied to tensors other than 2d matrices with row reduction.
    """
    
    __props__ = ("axis","keepdims")

    def __init__(self, axis=-1, keepdims=False):
        if not isinstance(axis, (integer_types, np.integer)):
            raise ValueError("axis must be a constant integer")
        self.axis = int(axis)
        self.keepdims = keepdims
        
    def make_node(self, input):
        input = tensor.as_tensor_variable(input)
        if input.type.ndim==0 or input.type.dtype not in tensor.float_dtypes:
            raise ValueError('input must be a >=1d tensor of floats. Got %s' %
                             input.type)

        # print >> sys.stderr, "*** in make_node input.type.dtype= ", input.type.dtype

        broadcastable = list(input.type.broadcastable)
        if self.keepdims:
            broadcastable[self.axis] = True
        else:
            del broadcastable[self.axis]
        broadcastable = tuple(broadcastable)

        output = TensorType(dtype=input.type.dtype,
                            broadcastable=broadcastable)()
        # return Apply(self, [input], [output])
        return Apply(self, [input], [output])

    def perform(self, node, input_storage, output_storage):
        x, = input_storage
        outputs, = output_storage
        result = logsumexp_npy(x, self.axis, self.keepdims)
        if not isinstance(result, np.ndarray): # workaround for test expecting ndarray, while numpy reduction returns scalar
            result = np.array(result)
        # print >> sys.stderr, "*** in perform: x.shape = ", x.shape, "result.shape = ", result.shape, "outputs = ",outputs
        outputs[0] = result

    def infer_shape(self, node, input_shapes):
        ishape, = input_shapes
        oshape = list(ishape)
        if self.keepdims:
            oshape[self.axis] = 1
        else:
            del oshape[self.axis]
        return [ tuple(oshape) ]
        
    def gradv1(self, inp, grads):
        x, = inp
        g, = grads
        if not self.keepdims:
            g = T.shape_padaxis(g,self.axis)
        return [g*softmax_with_axis(x,self.axis)]

    def gradv2(self, inp, grads):
        x, = inp
        g, = grads
        if not self.keepdims:
            g = T.shape_padaxis(g,self.axis)
        sm = T.exp(x-logsumexp(x,self.axis,keepdims=True))
        return [g*sm]


    def grad(self, inp, grads):
        return self.gradv2(inp,grads)

def logsumexp(x, axis=-1, keepdims=False):
    if isinstance(x, np.ndarray):
        return logsumexp_npy(x, axis, keepdims)
    else: # return theano op instance
        return LogSumExp(axis=axis, keepdims=keepdims)(x)

    
# @register_canonicalize
@register_stabilize
@register_specialize
@gof.local_optimizer([LogSumExp])
def local_LogSumExp_optim(node):
    # log(sum_i(exp(x_i))) = x_max + log(sum_i(exp(x_i - x_max)))

    if not isinstance(node.op,LogSumExp):
        return

    input = node.inputs[0]
    axis = node.op.axis
    keepdims = node.op.keepdims
    
    max = T.max(input, axis=axis)
    res = max + T.log( T.sum( T.exp( input-T.shape_padaxis(max,axis) ), axis=axis) )
    if node.op.keepdims:
        res = T.shape_padaxis(res,axis)

    return [res]


####  ######## WHAT FOLLOWS ARE OLD UNUSED FUNCTIONS AS WELL AS AN ABANDONED OPTIMISATION APPROACH ATTEMPT ##########
####  
####  def old_logsumexp_reducelast(A):
####  
####      if A.ndim==2:
####          # important: the following relies on Theano detecting that pattern and 
####          # performing the following numerical stabilization optimization:
####          #  - for backprop: notice that the gradient of this is softmax
####          #  - in the forward computation: numerically stabilize by subtracting the max computed along that axis, and adding it to the result
####          result = T.log(T.sum(T.exp(A), axis = 1))
####          
####          # own numerically stable version that subtracts the max (but dont know how the gradient will behave)
####          # maxA = A.max(axis=1, keepdims=True)
####          # result = maxA.squeeze() + T.log(T.sum(T.exp(A-maxA), axis = 1))
####  
####      else: 	# First flatten the leading dimensions, because softmax, which should be the grad of logsumexp, only detected/works on matrices
####          initial_shape = A.shape
####          lastdim = initial_shape[-1]
####          flatA = A.flatten()
####          Amat = flatA.reshape( (flatA.shape[0]/lastdim, lastdim) )
####          # result = T.log(T.sum(T.exp(Amat), axis = 1))
####          result = logsumexp_reducelast(Amat)
####          result = result.reshape(initial_shape[0:-1])
####  
####      return result
####  	
####  # TODO IMPORTANT FOR logsumexp_reducelast: verify that
####  #  a) theano's *gradient* of this T.log(T.sum(T.exp(A), axis=1)) gets correctly optimized to a Softmax(A) op
####  #  b) there is another optimization that detects this T.log(T.sum(T.exp(A), axis=last)) and optimizes it into
####  #      	A_max = T.max(A,axis=A.ndim-1)
####  #          A_reduced = T.sum(T.exp(A-T.shape_padright(A_max)),axis=A.ndim-1)
####  # 	     result = A_max + T.log(A_reduced)
####  
####  
####  def old_log_add_exp_elemwise(A,B):
####      if A is None:
####          return B
####      elif B is None:
####          return A
####  
####      ## own version using abs
####      # D = T.abs_(A-B)
####      # return (A+B+D)*0.5 + T.log1p(T.exp(-D))
####  
####      ## own version using max
####      # max_A_B = T.switch( A>B, A, B)
####      # return max_A_B + T.log(T.exp(A-max_A_B)+T.exp(B-max_A_B))
####  
####      ## softplus version
####      # return A+softplus(B-A)
####  
####      ## logsumexp_reducelast version
####      # return logsumexp_reducelast(T.stack((A,B),axis=-1))
####  
####      ## logsumexp_version
####      return logsumexp(T.stack((A,B),axis=-1))
####      
####  ######## WHAT FOLLOWS ARE UNUSED (CURRENTLY UNREGISTERED) OPTIMIZATIONS ##########
####          
####  # Alexandre's optimisation for detecting and stabilizing log(sum(exp(x),axis))
####  
####  # @register_canonicalize
####  #@register_stabilize
####  #@register_specialize
####  #@gof.local_optimizer([T.log])
####  def local_logsumexp(node):
####      # log(sum_i(exp(x_i))) = x_max + log(sum_i(exp(x_i - x_max)))
####  
####      print "@@@@@@ GLOUBIBOULGA @@@@@"
####      
####      if node.op != T.log:
####          return
####          
####      sum_node = node.inputs[0].owner
####      # If the sum has keepdims=True, there might be a dimshuffle
####      if sum_node and isinstance(sum_node.op, T.DimShuffle):
####          sum_node = sum_node.inputs[0].owner
####              
####      if not sum_node or not isinstance(sum_node.op, T.Sum):
####          return
####                  
####      exp_node, axis = sum_node.inputs[0].owner, sum_node.op.axis
####      if not exp_node and exp_node.op != T.exp:
####          return
####                      
####      pre_exp = exp_node.inputs[0]
####      # optimisation may have already been applied
####      
####      if (pre_exp.owner and
####          isinstance(pre_exp.owner.op, T.Elemwise) and
####          pre_exp.owner.op.scalar_op == scalar.sub):
####          max_node = pre_exp.owner.inputs[1].owner
####          if max_node and isinstance(max_node.op, T.DimShuffle):
####              max_node = max_node.inputs[0].owner
####          if not isinstance(max_node.op, T.MaxAndArgmax):
####              return
####          if max_node.inputs[0] == pre_exp.owner.inputs[0]:
####              return
####          
####      print "@@@@@@ YOUPI!!! @@@@@"                                    
####      max_pre_keepdims = T.max(pre_exp, axis=axis, keepdims=True)
####                                      
####      ret = (max_pre_keepdims + T.log(T.sum(T.exp(pre_exp - max_pre_keepdims),
####                                            axis=axis, keepdims=True)))
####                                      
####      # Restore shape and broadcastable pattern
####      ret = T.reshape(ret, node.inputs[0].shape)
####      ret = T.patternbroadcast(ret, node.inputs[0].broadcastable)
####                                      
####      return [ret]
####  
####  # Yet unsuccessful attempt at detecting and stabilizing  exp(x)/sum(exp(x),axis=axis,keepdims-True) by transforming it into a softmax along the specified axis
####  # inspired/adapted from softmax_simplifier in theano.tensor.nnet, trying to make it more general
####  # but strangely in my tests, it sdesn't seem to pick up the pattern: no DimShuffle in the denominator (while the initial softmax_simplifier did, at leas tin some cases...)
####  
####  def softmax_simplifier2(numerators, denominators):
####      print "YOUYOU: entering simplifier2!!!"
####      for numerator in list(numerators):
####          if not numerator.type.dtype.startswith('float'):
####              continue
####  
####          print "numerator op",numerator.owner.op
####          if numerator.owner and numerator.owner.op == tensor.exp:
####              x = numerator.owner.inputs[0]
####          else:
####              continue
####  
####          matching_denom = None
####  
####          print "n_denominaotrs",len(denominators)
####          for denominator in denominators:
####              print "simplifier2 considering denominator", denominator.owner.op, denominator.owner.inputs[0].owner.op
####              
####              dimshuffle_node = denominator.owner 
####              if not dimshuffle_node or not isinstance(dimshuffle_node.op,
####                                                       tensor.DimShuffle):
####                  # did not find expected dimshuffle (arising from from a sum with keepdims)
####                  continue 
####              sum_node = dimshuffle_node.inputs[0].owner 
####              if not isinstance(sum_node.op, tensor.Sum) or sum_node.inputs[0] is not numerator:
####                  # did not find expected sum of the numerator
####                  continue
####              if sum_node.op.axis is None or len(sum_node.op.axis)!=1:
####                  continue # not currently supporting more than a single axs
####              axis = sum_node.op.axis[0]
####              if type(axis) is not int:
####                  continue # softmax_with_axis supports only constant integer axis
####              expected_dimshuffle_order = arange(numerator.ndim)
####              expected_dimshuffle_order[axis] = 'x'
####              if expected_dimshuffle_order != list(dimshuffle.new_order): 
####                  # expected dimshuffle order did not match a keepdims sum
####                  continue
####              # OK all conditions satisfied. we found the matching denominator!
####              matching_denom = denominator
####              break    
####                  
####          if matching_denom:
####              print "YOUYOU: APPLYING MATCH simplifier2!!!"
####              softmax = softmax_with_axis(x,axis=axis)
####              copy_stack_trace(numerator, softmax)
####              numerators.remove(numerator)
####              denominators.remove(matching_denom)
####              numerators.append(softmax)
####  
####      return numerators, denominators
####  
####  #opt.local_mul_canonizer.add_simplifier(softmax_simplifier2, 'softmax_simplifier2')
####  
