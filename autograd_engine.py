import numpy as np
import logging

'''
 NEED TO ADD OVERLOAD FOR MATMUL
'''

logging.basicConfig(filename='backward.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.array(data, dtype=np.float64) 
        self.shape = self.data.shape
        

        self.grad = np.zeros_like(self.data) if requires_grad else None

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)
        for child in self._prev:
            child.zero_grad()

    def _preprocess_binop(self, other):
        '''
        Inspiration: https://github.com/smolorg/smolgrad/blob/master/smolgrad/core/engine.py
        to process for binary operations, like add, mul, sub etc

        '''
        other = other if isinstance(other, Tensor) else Tensor(other)
        
 
        if self.shape == () or other.shape == ():
            return self, other

  
        try:
            broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")

        self_broadcasted = self.broadcast_to(broadcast_shape)
        other_broadcasted = other.broadcast_to(broadcast_shape)
        
        return self_broadcasted, other_broadcasted

    def broadcast_to(self, shape):
        if self.shape == shape:
            return self
        
        data = np.broadcast_to(self.data, shape)
        out = Tensor(data, _children=(self,), _op="broadcast", requires_grad=self.requires_grad)
        
        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if out.grad is None:
                return
            
            '''
            get the gradient of out and find axes that was broadcasted
            find dimensions for self and out

            logic: if oroginal dimension is 1 but the new dimension is not, it was broadcasted along that axis
            these axes are saved

            then checks if anything is in the broadcasted axes

            if there is, it sums over the new axes

            for example if we broadcast a tensor from shape (3, 1) to (3, 4) 
            then the single value in the 2nd dim is repeated 4 times and hence sum up the gradients across all 4
            corresponding positions in the larger tensor 
            AS IT IS BACKWARDS

            not forgetting to reshape the self grad etc...

            '''
            grad = out.grad
            
            broadcast_axes = []
            for i, (dim_out, dim_self) in enumerate(zip(shape, self.shape)):
                if dim_self == 1 and dim_out != 1:
                    broadcast_axes.append(i)
           
            if broadcast_axes:
                grad = grad.sum(axis=tuple(broadcast_axes), keepdims=True)
            self.grad += grad.reshape(self.shape)
        
        out._backward = _backward
        return out

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data < other)
        elif isinstance(other, Tensor):
            return Tensor(self.data < other.data)
        else:
            raise TypeError(f"Unsupported operand type for <: '{type(self)}' and '{type(other)}'")

    def view(self, *shape):
        '''
        simple implenetation of view (to reshape)
        w/ support for -1 in shape
        for example if tensor is 2, 3 shape 
        usinfg view(3, 2) would shape it to 3, 2
        '''
        
        if -1 in shape:
            
            known_dim_product = 1
            unknown_dim_index = -1
            for i, dim in enumerate(shape):
                if dim == -1:
                    if unknown_dim_index != -1:
                        raise ValueError("Can only specify one unknown dimension")
                    unknown_dim_index = i
                else:
                    known_dim_product *= dim
            
            
            total_elements = np.prod(self.shape)
            if total_elements % known_dim_product != 0:
                raise ValueError(f"Cannot reshape tensor of shape {self.shape} to shape {shape}")
            
            unknown_dim = total_elements // known_dim_product
            shape = list(shape)
            shape[unknown_dim_index] = unknown_dim
            shape = tuple(shape)
        
        
        if np.prod(shape) != np.prod(self.shape):
            raise ValueError(f"Cannot reshape tensor of shape {self.shape} to shape {shape}")
        
        reshaped_data = self.data.reshape(shape)
        out = Tensor(reshaped_data, _children=(self,), _op='view')
        
        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if out.grad is None:
                return
            
            self.grad += out.grad.reshape(self.shape)
        
        out._backward = _backward
        return out


    def __add__(self, other):
        '''
        uses _preprocess binop
        '''
        self_broadcasted, other_broadcasted = self._preprocess_binop(other)
        out = Tensor(
            self_broadcasted.data + other_broadcasted.data,
            _children=(self_broadcasted, other_broadcasted),
            _op="+",
            requires_grad=self.requires_grad or other_broadcasted.requires_grad
        )
        
        def _backward():
            if out.grad is None:
                return
            
            if self_broadcasted.requires_grad:
                if self_broadcasted.grad is None:
                    self_broadcasted.grad = np.zeros_like(self_broadcasted.data)
                self_broadcasted.grad += out.grad
            
            if other_broadcasted.requires_grad:
                if other_broadcasted.grad is None:
                    other_broadcasted.grad = np.zeros_like(other_broadcasted.data)
                other_broadcasted.grad += out.grad
        
        out._backward = _backward
        return out




    def __mul__(self, other):
        self_broadcasted, other_broadcasted = self._preprocess_binop(other)
        out = Tensor(
            self_broadcasted.data * other_broadcasted.data,
            _children=(self_broadcasted, other_broadcasted),
            _op="*",
            requires_grad=self.requires_grad or other_broadcasted.requires_grad
        )
        
        def _backward():
            if out.grad is None:
                return
            
            if self_broadcasted.requires_grad:
                if self_broadcasted.grad is None:
                    self_broadcasted.grad = np.zeros_like(self_broadcasted.data)
                self_broadcasted.grad += other_broadcasted.data * out.grad
            
            if other_broadcasted.requires_grad:
                if other_broadcasted.grad is None:
                    other_broadcasted.grad = np.zeros_like(other_broadcasted.data)
                other_broadcasted.grad += self_broadcasted.data * out.grad
        
        out._backward = _backward
        return out



    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        self_broadcasted, other_broadcasted = self._preprocess_binop(other)
        out = Tensor(
            self_broadcasted.data - other_broadcasted.data,
            _children=(self_broadcasted, other_broadcasted),
            _op="-",
            requires_grad=self.requires_grad or other_broadcasted.requires_grad
        )
        
        def _backward():
            if out.grad is None:
                return
            
            if self_broadcasted.requires_grad:
                if self_broadcasted.grad is None:
                    self_broadcasted.grad = np.zeros_like(self_broadcasted.data)
                self_broadcasted.grad += out.grad
            
            if other_broadcasted.requires_grad:
                if other_broadcasted.grad is None:
                    other_broadcasted.grad = np.zeros_like(other_broadcasted.data)
                other_broadcasted.grad -= out.grad
        
        out._backward = _backward
        return out



    def expand(self, *shape):
        '''
        expands a tensor to new shape, obviously cant reduce. 
        the backprop obviously goes backwards so from expanded shape to smaller shape and hence adds over
        the expanded dims
        '''
        if len(shape) < len(self.shape):
            raise ValueError(f"Expanded shape {shape} must have at least {len(self.shape)} dimensions")
        
        new_shape = list(shape)
        for i, (new_dim, old_dim) in enumerate(zip(reversed(shape), reversed(self.shape))):
            if new_dim != old_dim and old_dim != 1:
                raise ValueError(f"Expanded shape {shape} is not compatible with current shape {self.shape}")
            if old_dim == 1:
                new_shape[-(i+1)] = new_dim
        
        expanded_data = np.broadcast_to(self.data, new_shape)
        out = Tensor(expanded_data, _children=(self,), _op='expand')
        
        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if out.grad is None:
                return
            
            
            grad = out.grad
            broadcast_axes = []
            for i, (dim_out, dim_self) in enumerate(zip(new_shape, self.shape)):
                if dim_self == 1 and dim_out != 1:
                    broadcast_axes.append(i)
            if broadcast_axes:
                grad = grad.sum(axis=tuple(broadcast_axes), keepdims=True)
            self.grad += grad.reshape(self.shape)
        
        out._backward = _backward
        return out

    def pow(self, exponent):
        requires_grad = self.requires_grad
        out = Tensor(np.power(self.data, exponent), (self,), 'pow', requires_grad=requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += (exponent * np.power(self.data, exponent - 1)) * out.grad
        
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        return self.pow(exponent)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data / other.data, (self, other), '/', requires_grad=requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (1 / other.data) * out.grad

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                self_over_other_squared = self.data / (other.data ** 2 + 1e-8)  # Added epsilon for stability
                other.grad += (-self_over_other_squared) * out.grad
    
        out._backward = _backward
        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data + 1e-8), (self,), 'sqrt', requires_grad=self.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += 0.5 / np.sqrt(self.data + 1e-8) * out.grad
    
        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(np.matmul(self.data, other.data), (self, other), 'matmul', requires_grad=requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.matmul(out.grad, other.data.T)
            
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += np.matmul(self.data.T, out.grad)
    
        out._backward = _backward
        return out


    def sum(self, axis=None, keepdims=False):
        
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum', requires_grad=self.requires_grad)
        
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            
            if axis is None:
                
                self.grad += np.ones_like(self.data) * out.grad
            else:
                
                expanded_grad = np.expand_dims(out.grad, axis=axis)
                self.grad += expanded_grad * np.ones_like(self.data)
    
        
        out._backward = _backward
        
        return out

    def tanh(self):
        '''
        included because useful for acrtivation
        '''
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh', requires_grad=self.requires_grad)
        
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            # Accumulate gradient
            self.grad += (1 - t**2) * out.grad
        
       
        out._backward = _backward
        
        return out 

    def backward(self, gradient=None):
        '''
        the magic part. uses the topo sort
        builds based off the parents children to go backwards
        porpragtes chain rule

        the reverse has logging enabled for troubleshooting
        can comment out if not satisfactory
        '''

        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
        
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise RuntimeError("grad must be specified for non-scalar tensor")
        
        self.grad = gradient
        
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        
        for v in reversed(topo):
            logging.info(f"Processing tensor with shape {v.shape}, op '{v._op}'")
            v._backward()
            if v.grad is not None:
                logging.info(f"After _backward, grad shape: {v.grad.shape}")
            else:
                logging.info("After _backward, grad is None")




    def __getitem__(self, idx):
        '''
        used for indexing
        '''
        if isinstance(idx, Tensor):
            idx = idx.data.astype(int)  
        elif isinstance(idx, np.ndarray) and idx.dtype != np.int64:
            idx = idx.astype(int)  

        out = Tensor(self.data[idx], _children=(self,), _op='getitem')
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
    
            
            grad = np.zeros_like(self.data)
            np.add.at(grad, idx, out.grad)
            self.grad += grad
    
        out._backward = _backward
        return out

    
    def reshape(self, *shape):
        '''
        works for non continguous tensors 
        see strides for more info:https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays

        '''
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape', requires_grad=self.requires_grad)
        
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            
            self.grad += out.grad.reshape(self.shape)
    
        out._backward = _backward
        return out


    def transpose(self, *axes):

        '''
        classic transpose

        uses argsort for the inverse backwards and re transpsoes accoriidnlgy
        '''
        if not axes:
            axes = tuple(reversed(range(len(self.shape))))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        
       
        if len(axes) < len(self.shape):
            remaining_axes = [i for i in range(len(self.shape)) if i not in axes]
            axes = list(axes) + remaining_axes
        
        if len(axes) != len(self.shape):
            raise ValueError(f"Invalid number of axes. Got {len(axes)}, expected {len(self.shape)}")
        
        out = Tensor(self.data.transpose(*axes), (self,), 'transpose', requires_grad=self.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            
          
            inverse_axes = np.argsort(axes)
            self.grad += out.grad.transpose(*inverse_axes)
        
        out._backward = _backward
        return out


    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), 'relu', requires_grad=self.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp', requires_grad=self.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out


    def unsqueeze(self, dim):
        '''
        add extra dim
        '''
        new_shape = list(self.data.shape)
        new_shape.insert(dim, 1)
        out = Tensor(self.data.reshape(new_shape), (self,), 'unsqueeze', requires_grad=self.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            
            self.grad += out.grad.reshape(self.data.shape)
        
        out._backward = _backward
        return out



    def squeeze(self, dim=None):
        '''
        remove extra dim
        '''
        if dim is None:
            new_shape = tuple(s for s in self.data.shape if s != 1)
        else:
            new_shape = list(self.data.shape)
            if new_shape[dim] == 1:
                new_shape.pop(dim)
            else:
                raise ValueError(f"Cannot squeeze dimension {dim} as its size is not 1.")
        
        out = Tensor(self.data.reshape(new_shape), (self,), 'squeeze', requires_grad=self.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            
            self.grad += out.grad.reshape(self.data.shape)
        
        out._backward = _backward
        return out


    def concatenate(self, tensors, axis=0):
        '''
        concat op
        '''
        tensors = [self] + [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        
        shape_check = [t.shape[:axis] + t.shape[axis + 1:] for t in tensors]
        if not all(s == shape_check[0] for s in shape_check):
            raise ValueError("All tensors must have the same shape except in the concatenation axis.")
        
        data = np.concatenate([t.data for t in tensors], axis=axis)
        out = Tensor(data, _children=tensors, _op='concatenate', requires_grad=self.requires_grad or any(t.requires_grad for t in tensors))
    
        def _backward():
            if out.grad is None:
                return  
            
            splits = np.cumsum([t.shape[axis] for t in tensors[:-1]])
            split_gradients = np.split(out.grad, splits, axis=axis)
            
            for t, grad in zip(tensors, split_gradients):
                if t.requires_grad:
                    if t.grad is None:
                        t.grad = np.zeros_like(t.data)
                    t.grad += grad
    
        out._backward = _backward
        return out


    def split(self, num_splits, axis=-1):
        '''
        this splits it, hence called split
        '''
        splits = np.array_split(self.data, num_splits, axis=axis)
        return [Tensor(split, _children=(self,), _op='split') for split in splits]



    def norm(self, axis=None, keepdims=False):
        
        '''
        norm uses epsilon of 1e-8 for numerical stability
        need to find better approaches


        '''
        squared = self * self  
        summed_tensor = squared.sum(axis=axis, keepdims=keepdims)
        norm_data = np.sqrt(summed_tensor.data + 1e-8)         
        out = Tensor(norm_data, (squared,), 'norm', requires_grad=self.requires_grad)
        

        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            
        
            scale = self.data / (out.data + 1e-8)
            
            if axis is None:
                
                expanded_grad = out.grad  
            else:
                expanded_grad = out.grad
                if not isinstance(axis, tuple):
                    axis_tuple = (axis,)
                else:
                    axis_tuple = axis
                for ax in sorted(axis_tuple):
                    expanded_grad = np.expand_dims(expanded_grad, axis=ax)
                expanded_grad = np.broadcast_to(expanded_grad, self.data.shape)
            
            self.grad += scale * expanded_grad

        out._backward = _backward
        return out


    def cross(self, other):
        assert self.data.shape[-1] == 3 and other.data.shape[-1] == 3, "Cross product is only defined for 3D vectors"
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(np.cross(self.data, other.data), _children=(self, other), _op='cross', requires_grad=requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.cross(out.grad, other.data)
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += np.cross(self.data, out.grad)
        
        out._backward = _backward
        return out


    def outer(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(np.outer(self.data, other.data), (self, other), 'outer', requires_grad=requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
              
                self.grad += np.sum(out.grad * other.data, axis=1)
            
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
               
                other.grad += np.sum(out.grad * self.data[:, np.newaxis], axis=0)
        
        out._backward = _backward
        return out

'''

below scatter mean and scatter sum used in my MPNN implementation

they both aggregate values from the source tensor based on the index

computes the respective mean or sum of these indexes, in the case of the MPNN for the distances
'''

def scatter_mean(src, index, dim_size):
    if isinstance(index, Tensor):
        index = index.data.astype(np.int64)  
    else:
        index = index.astype(np.int64)
    
    if len(src.shape) > 2:
    
        src_2d = src.view(-1, src.shape[-1])
    else:
        src_2d = src
    out = Tensor(np.zeros((dim_size, src_2d.shape[-1])))
    count = np.zeros(dim_size)
    np.add.at(out.data, index, src_2d.data)
    np.add.at(count, index, 1)
    count[count == 0] = 1  
    out.data /= count[:, None]
    

    def _backward():
        if out.grad is None:
            return
        if src.grad is None:
            src.grad = np.zeros_like(src.data)
        
     
        src.grad += out.grad[index] / count[index][:, None]
    

    out._backward = _backward
    return out


def scatter_sum(src, index, dim_size):
    if isinstance(index, Tensor):
        index = index.data.astype(np.int64)
    else:
        index = index.astype(np.int64)

    if len(src.shape) > 2:
        src_2d = src.view(-1, src.shape[-1]) 
    else:
        src_2d = src
    out = Tensor(np.zeros((dim_size, src_2d.shape[-1])))
    np.add.at(out.data, index, src_2d.data) 

    def _backward():
        if out.grad is None:
            return
        if src.grad is None:
            src.grad = np.zeros_like(src.data)


        src.grad += out.grad[index]

    out._backward = _backward
    return out

'''
helper functions
'''
def tensor(data):
    return Tensor(data)

def randn(*shape):
    return Tensor(np.random.randn(*shape))

def zeros(*shape):
    return Tensor(np.zeros(shape))

