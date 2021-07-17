"""
HicksPy -- distributed sources and receivers in FD.

(c) 2019-2021 Kajetan Chrapkiewicz.
Copywright: Ask for permission writing to k.chrapkiewicz17@imperial.ac.uk.

"""
from autologging import logged, traced
import matplotlib.pyplot as plt
import numpy as np
import time 
clip_at = 1e-9  # spread-factors smaller than this will be neglected


def timer(func):
  """
  Print the time taken to run 
  the decorated function.
  
  """
  @wraps(func)
  def wrapper_timer(*args, **kwargs):
    timer = kw('timer', False, kwargs)
    
    start_time = time.perf_counter()
    value = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    if timer:
      t = end_time - start_time
      print('Exec. time: {} s <{}>'.format("{:15.12f}".format(t), func.__name__))    
    return value
  return wrapper_timer


@traced
@logged
def kaiser(x, r=3, dipole=False, **kwargs):
  """
  Value of the Kaiser-windowing function at point x.
  Bessel function can computed using different 
  implementations.
  
  Parameters
  ----------
  b: float
    Shape of window
  r: float
    Radius of window
  bessel : str 
    Implementation to choose: fw3D or python built-in.
  x : float
    Distance from the centre of sinc. 
    
  Returns
  -------
  value : float
    Value of the Kaiser-windowed sinc function at point x.
  
  Notes 
  -----
  See Hicks 2002 for details.
  
  It can be vectorized because 'if' statement was replaced 
  with np.where.
  
  """
  from scipy.special import i0
  
  assert isinstance(r, int)
  assert r <= 5 
  
  # OPTIMAL VALUES FOR r=0 (-> None), r=1, ... FROM Hicks2002/Table 1 & 2
  b_optim_mono = [None, 0.00, 1.84, 3.04, 4.14, 5.26] # kmax = 2/3 pi
  b_optim_dipo = [None, 0.00, 1.48, 3.25, 4.40, 5.44] # kmax = 2/3 pi 
  
  if dipole:
    b = kw('b', b_optim_dipo[r], kwargs)
  else:
    b = kw('b', b_optim_mono[r], kwargs)

  #if r != 3:
    #kaiser._log.debug('Using r=%s, not 3 as in Fullwave3D.' % r)
  
  #if r == 3 and b != 4.14:
    #kaiser._log.debug('Using b=%s, not 4.14 as in Fullwave3D.' % b)
  
  bessel = 'py'
  if bessel == 'py':
    bessel = lambda x : i0(x)
  elif bessel == 'fw3d':
    bessel = lambda x : bessel_fw3d(x)
  else:
    raise ValueError('Unknown implementation of the Bessel function: ' + bessel)
  
  frac = (x / r) ** 2
  return np.where(frac > 1, 0.0, bessel(b * np.sqrt(1 - frac)) / bessel(b))
@traced
@logged
def sinc(x, **kwargs):
  """
  Calculate value of the sinc function defined as:
         sinc(x) = sin(pi * x) / (pi * x).
  
  Parameters
  ----------
  x : float
    Argument of the sinc(x); Any real number.
  
  Returns
  -------
  value : float
    Value of the sinc(x).
    
  Notes
  -----
  Definition containing pi number is favorable 
  because in this case Sinc has its roots at integer 
  numbers (finite-difference nodes), not pi-multiples.
  
  Numerical stability is addressed.
  
  """
  x = np.array(x)
  # ALLOWS VECTORIZATION WITH CONDITION INSIDE
  return np.where(abs(x) < epsi, 1.0, np.sin(pi * x) / (pi * x)) 
@traced
@logged
def dsinc_dx(x, **kwargs):
  """
  sinc'(x)
  
  Notes
  -----
  Unlike sinc:
  - max > 1
  - odd function
  
  See sinc docs too.
  
  """
  x = np.array(x)
  # NOTE dsinc_dx(0) = 0
  return np.where(abs(x) < epsi, 0.0, (np.cos(pi * x) - sinc(x)) / x)
@traced
@logged
class Plotter(object):
  """
  Generic plotting mix-in. It wrapps the plot method of child classes.
  The rationale is that _initalize, _finalize and _save should be the same,
  regardless of the object properties to be plotted.
   
  If really needed in special cases, fine-grained customization 
  should done in the plot method of child classes. Note that they can 
  put the both 
  """
  def overlay(self, *layers, **kwargs):
    """
    Actually, we should develop this one as last, since it can 
    be achieved easily in the notebook.
    """
    self.plott(**kwargs)
    for layer in layers:
      layer.plot(**kwargs) # not plott, otherwise new figure

  # -----------------------------------------------------------------------------

  def plott(self, *args, **kwargs):
    """
    We use a different name (plott) to stay compatible with 
    plot methods of child classes.
    """
    save = kwargs.get('save', True)
    kwargs = self._initialize(**kwargs)
    ax = self.plot(*args, **kwargs)
    ax = self._finalize(**kwargs)
    if save:
      self._save(**kwargs)

  # -----------------------------------------------------------------------------

  def _initialize(self, **kwargs):
    """
    Set everything that needs to be set before the actual plotting
    function (imshow etc.) is called.
    """
    figure(**kwargs)
    return kwargs
  
  # -----------------------------------------------------------------------------

  def plot(self, **kwargs):
    raise NotImplementedError('Overwritten in a child class.')

  # -----------------------------------------------------------------------------

  def _finalize(self, grid={}, **kwargs):
    """
    Add final formatting.
    """
    assert isinstance(grid, dict)
    if len(grid) > 0:
      plt.grid(**grid)

  # -----------------------------------------------------------------------------

  def _save(self, fname, fmt='png', close=True, **kwargs):
    fname = strip(fname)
    fname = '%s.%s' % (fname, fmt)
    plt.savefig(fname)
    if close:
      plt.close()

  # -----------------------------------------------------------------------------
@traced
@logged
class Arr(np.ndarray):
  """
  Wrapper around numpy's array.
  
  """
  def __new__(cls, source, ndims=None, **kwargs):
    """
    Init by reading from source.
    
    Notes
    -----
    From https://docs.scipy.org/doc/numpy/user/basics.subclassing.html:
    Input array is an already formed ndarray instance

    """
    if hasattr(source, 'extent'): # NOTE ADDED 12.01.2021
      kwargs['extent'] = source.extent 

    source = cls._read(source, **kwargs)
    
    obj = np.asarray(source).view(cls) # CAST THE TYPE
    # FIXME: REPLACE IT WITH STH TAKING source 
    # AS ARG AND RETURNING EXTENT WHICH WE'LL ASSIGN TO obj JUST BEFORE RETURNING IT
    # FROM THIS __new__ FUNCTION
    
    obj = cls._set_extent(obj, **kwargs) 
        

    #obj = cls._set_coords(obj, **kwargs)
    obj = cls._set_dx(obj, **kwargs)

    if ndims is not None:
      assert len(obj.shape) == ndims
    
    return obj # NECESSARY!

  # -----------------------------------------------------------------------------
 
  def _read(source, **kwargs):
    """
    """
    #from fullwavepy.seismic.data import Data
    #from fullwavepy.ndat.manifs import Surf, SurfZ, Plane
    #from fullwavepy.ndat.points import Points
    #
    #if (type(source) == type(np.array([])) or 
    #    type(source) == Arr or
    #    type(source) == Arr1d or
    #    type(source) == Arr2d or
    #    type(source) == Arr3d or
    #    type(source) == Data or
    #    type(source) == Surf or
    #    type(source) == SurfZ or
    #    type(source) == Plane or
    #    type(source) == Points or
    #    type(source) == np.memmap):
    #  A = source    

    if isinstance(source, str):
      from fullwavepy.ioapi.generic import read_any
      if hasattr(source, 'shape'): # FOR EFFICIENCY (SEE read_any)
        kwargs['shape'] = self.shape
        
      A = read_any(source, **kwargs)
    else:
      A = source


    #else:
    # raise TypeError('Arguments need to be either ' + 
    #                 'file-names or arrays or np.memmap, NOT: %s' %
    #                 type(source))
    return A
  
  # -----------------------------------------------------------------------------   
  
  def _set_extent(obj, func=None, **kwargs):    
    if 'extent' in kwargs:
      obj.__log.debug('Using extent from kwargs, even if it means overwriting')
      obj.extent = kwargs['extent']
    
    elif hasattr(obj, 'extent'):
      obj.__log.debug('obj.extent already set and not provided in kwargs')
      pass
    
    else:
      obj.__log.debug('Setting extent to default.')
      obj.extent = obj._default_extent(func, **kwargs)
    
    return obj
  
  # -----------------------------------------------------------------------------  
  
  def _default_extent(obj, func=None, **kwargs):
    """
    Redefined in child classes to account for vertical axis flipping 
    when plotting with imshow.
    
    """
    if func is None:
      func = lambda dim : [0, dim-1]   # outdated: # NOT dim-1; SEE GridProjFile ETC.
    extent = []
    for dim in obj.shape:
      extent.append(func(dim))
    
    # if len(obj.shape) == 1:
    #   extent = extent[0]
    
    return extent
  
  # -----------------------------------------------------------------------------

  def _set_dx(obj, **kwargs):
    """
    It is fully determined by extent and shape.
    In general, it is axis-dependent (dx != dy != dz != dx)
    """
    dx = []
    obj.__log.debug('obj.shape %s' % str(obj.shape))
    obj.__log.debug('obj.extent %s' % str(obj.extent))
    assert len(obj.shape) == len(obj.extent)
    for nx, (x1, x2) in zip(obj.shape, obj.extent):
      obj.__log.debug('nx=%s, x1=%s, x2=%s' % (nx, x1, x2))
      dx_1D = (x2 - x1) / (nx-1) if nx > 1 else None
      obj.__log.debug('dx_1D=%s' % dx_1D)
      dx.append(dx_1D)
    
    obj.dx = np.array(dx)
    return obj

  # -----------------------------------------------------------------------------
  
  def _set_coords(obj, **kwargs):
    obj.__log.debug('obj.extent' + str(obj.extent))
    obj.__log.debug('Setting coords to None. Fill it with actual code')
    obj.coords = None
    
    return obj
  
  #def _set_shape(obj, shape=None, **kwargs):
    #self.shape = None
   
  # -----------------------------------------------------------------------------

  def __array_finalize__(self, obj):
    if obj is None: return
  
  # -----------------------------------------------------------------------------  

  def _metre2index(self, m, axis, **kwargs):
    origin = self.extent[axis][0]
    i = (m - origin) / self.dx[axis]
    if not i.is_integer():
      raise ValueError('Index must be integer not %s' % i)
    return int(i)

  # -----------------------------------------------------------------------------  

  def _metre_2_nearest_index(self, m, axis, **kwargs):
    """
    Better version of _metre2index used 
    by fwilight.ndat.A3d and A2d. 

    Parameters
    ----------
    m : float
        Value in metres.
    axis : int
        Axis of the array.

    Returns
    -------
    int
        Nearest index.
    """
    origin = self.extent[axis][0]
    i = (m - origin) / self.dx[axis]
    if not i.is_integer():
      print('Warning. Non-integer index. Taking its floor')
      i = np.floor(i)
    return int(i) 

  # -----------------------------------------------------------------------------  

  def _index2metre(self, i, axis, **kwargs):
    origin = self.extent[axis][0]
    m = i * self.dx[axis] + origin
    return m

  # -----------------------------------------------------------------------------  
  
  def _metre2gridnode(self, *args, **kwargs):
    return self._metre2index(*args, **kwargs) + 1  
  
  # -----------------------------------------------------------------------------

  def _box2inds(self, box, **kwargs):
    """
    Convert box into slicing-indices using extent.
    
    """
    box = np.array(box)
    extent = np.array(self.extent)
    assert len(box.shape) == 1
    assert len(box) == len(extent.flatten())
    box = box.reshape(extent.shape)
    inds = np.zeros(box.shape)
    for axis, _ in enumerate(box):
      b0, b1 = box[axis]
      if b0 == b1: # FOR 2D (DOUBLE-CHECK)
        self.__log.warn('Skipping b0=b1=%s' % b0)
        continue
      inds[axis][0] = self._metre2index(b0, axis)
      inds[axis][1] = self._metre2index(b1, axis) + 1 # NOTE: FOR np.arange(b1, b2) etc.
      self.__log.debug('axis %s: i1=%s, i2=%s' % (axis, inds[axis][0], inds[axis][1]))    
    return inds.astype(int)

  # ----------------------------------------------------------------------------- 

  def carve(self, box, **kwargs):
    """
    Carve a box out of an array.
    
    Parameters
    ----------
    box : list

    Returns
    -------
    self

    """
    inds = self._box2inds(box, **kwargs)
    
    for axis in range(len(self.shape)):
      self = np.take(self, np.arange(*inds[axis]), axis=axis)
    
    self.extent = np.array(box).reshape(inds.shape)
    return self
  
  # -----------------------------------------------------------------------------  
 
  def save(self, fname, **kwargs):
    from fullwavepy.ioapi.fw3d import save_vtr
    save_vtr(self, fname)
  
  # -----------------------------------------------------------------------------

  def info(self, **kwargs):
    self.__log.info('grid shape: {} [nodes]'.format(self.shape))
    self.__log.info('grid cell-sizes in (x,y,z): {} [m]'.format(self.extent))    
    self.__log.info('grid extent: {} [m]'.format(self.extent))
    self.__log.info('value min: {}, max: {}'.format(np.min(self), np.max(self)))

  # -----------------------------------------------------------------------------
  
  ###@widgets()
  def compare(self, othe, mode='interleave', **kwargs): #fig, gs=None, widgets=False, 
    if mode == 'interleave' or mode == 'ileave':
      A = self.interleave(othe, **kwargs)
      A.plot(**kwargs)
    # elif mode == 'diff' or mode == 'dif':

    #   c = A3d(self-othe, extent=self.extent)
    #   c.plot(**kwargs)
    #   return c

    else:
      raise ValueError(mode)
    
  # -----------------------------------------------------------------------------
  
  def compare_subplots(self, **kwargs):
      assert type(self) == type(othe)
      assert self.shape == othe.shape
      
      xlim = kw('xlim', None, kwargs)
      ylim = kw('ylim', None, kwargs)
      
      if widgets:
        figsize = (kw('figsize_x', 8, kwargs), kw('figsize_y', 8, kwargs))
        fig = plt.figure(figsize=figsize)
        kwargs['widgets'] = False
      
      if gs is None:
        gs = fig.add_gridspec(1,2)    
      
      
      ax1 = fig.add_subplot(gs[0,0])
      self.plot(**kwargs)
      ax2 = fig.add_subplot(gs[0,1])
      othe.plot(**kwargs)
      
      for ax in [ax1, ax2]:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
@traced
@logged
class Arr3d(Plotter, Arr):
  """
  3D array.
  
  """
  def __new__(cls, source, **kwargs):
    return super().__new__(cls, source, ndims=3, **kwargs)
  
  # -----------------------------------------------------------------------------
  
  ###@widgets('slice_at', 'node')
  def slice(self, slice_at='y', node=0, widgets=False, **kwargs):
    """
    """
    di = {'x': 0, 'y': 1, 'z': 2} # TRANSLATE slice_at INTO AXIS NO.
    axis = di[slice_at]
    A = Arr2d(np.take(self, indices=node, axis=axis))
    
    assert len(self.extent) == 3
    # extent2d = np.ravel([el for i, el in enumerate(self.extent) if i != di[slice_at]])
    extent2d = np.array([el for i, el in enumerate(self.extent) if i != di[slice_at]])

    # if axis != 2:
    self.__log.debug('Setting extent2d so that no vertical-axis flipping is needed.')
    self.__log.debug('NOW ALSO FOR zslice (NOT TESTED BUT SEEMS TO HAVE FIXED THE BUG)')
    # extent2d[-2: ] = [extent2d[-1], extent2d[-2]]
    extent2d[-1] = extent2d[-1][::-1]
    self.__log.debug('extent2d: ' + str(extent2d))
    
    A.extent = extent2d
    
    return A
  
  # -----------------------------------------------------------------------------
  
  ###@widgets('chunk_size')
  def interleave(self, othe, *args, **kwargs):
    A1 = self.slice(*args, **kwargs)
    A2 = othe.slice(*args, **kwargs)
    A = Arr2d(interleave_arrays(A1, A2, **kwargs))
    return A

  # -----------------------------------------------------------------------------
  
  ##@widgets('cmap', 'slice_at', 'node')
  def plot_slice(self, slice_at='y', node=None, widgets=False, **kwargs):
    """
    """
    nx, ny, nz = self.shape
    if node is None:
      if slice_at == 'x':
        node = kw('node', nx//2, kwargs)
        # metre = self._index2metre(node, 0)
      elif slice_at == 'y':
        node = kw('node', ny//2, kwargs)
        # metre = self._index2metre(node, 1)
      elif slice_at == 'z':
        node = kw('node', nz//2, kwargs) 
        # metre = self._index2metre(node, 2)     
      else:
        raise ValueError('Wrong slice_at: %s' % str(slice_at))

    arr2d = self.slice(slice_at, node, widgets=False, **kwargs)
    suffix = kwargs.get('title', '')
    suffix = ', ' + suffix if suffix != '' else suffix
    kwargs['title'] = 'Array slice at %s-index %s%s' % (slice_at, node, suffix)
    del_kw('slice_at', kwargs) # JUST IN CASE
    
    ax = arr2d.plot(**kwargs)
  
    if slice_at == 'z': # DISABLE?
      ax.invert_yaxis()
    return ax

  # -----------------------------------------------------------------------------

  def plot_3slices_new2(self, x, y, z, fig=None, gs=None, **kwargs):
    """
    """
    from fullwavepy.plot.plt2d import plot_image
    layout = kw('layout', 'square', kwargs)
 

    if fig is None:
      fig = figure(16,8)
    
    kwargs['x'] = x
    kwargs['y'] = y
    kwargs['z'] = z

    # LABELS FOR EACH AXIS
    s2 = kw('slice', 'y', kwargs) # MAIN SLICE PLOTTED AT THE BOTTOM IN FULL WIDTH
    s0, s1 = [i for i in ['x', 'y', 'z'] if i != s2]
    s = [s0, s1, s2]
    # CONVERT THE LABELS INTO ARRAY DIMENSIONS (AXES)
    convert_s2a = {'x': 0, 'y': 1, 'z': 2} # TRANSLATE slice TO axis
    

    if layout == 'square':
      if gs is None:
        gs = GridSpec(2,2, height_ratios=[1,1], width_ratios=[2,1])
      axes = list(np.zeros(3))
      axes[0] = fig.add_subplot(gs[0,0])
      axes[1] = fig.add_subplot(gs[1,0])
      axes[2] = fig.add_subplot(gs[:,1]) 
    elif layout == 'thin':
      if gs is None:
        gs = GridSpec(3,1)
      axes = list(np.zeros(3))
      axes[0] = fig.add_subplot(gs[0,0])
      axes[1] = fig.add_subplot(gs[1,0])
      axes[2] = fig.add_subplot(gs[2,0])   
    else:
      raise ValueError('Unknown layout: %s' % layout)  


    kwargs['vmin'] = kw('vmin', np.min(self), kwargs)
    kwargs['vmax'] = kw('vmax', np.max(self), kwargs)
    self.__log.debug('Setting vmin, vmax to: {}, {}'.format(kwargs['vmin'], 
                                                            kwargs['vmax']))
    
    for i, ax in enumerate(axes):
      plt.sca(ax)
      aaxx = plot_image(np.take(self, kwargs[s[i]], convert_s2a[s[i]]), **kwargs)
      aspeqt(aaxx)

      # PLOT SLICING LINES
      a, b = [j for j in ['x', 'y', 'z'] if j != s[i]]
      abcissae_horiz = range(self.shape[convert_s2a[a]])
      ordinate_horiz = np.full(len(abcissae_horiz), kwargs[b])
      ordinate_verti = range(self.shape[convert_s2a[b]])
      abcissae_verti = np.full(len(ordinate_verti), kwargs[a])
      
      if s[i] == 'z':
        abcissae_horiz, ordinate_horiz, abcissae_verti, ordinate_verti = abcissae_verti, ordinate_verti, abcissae_horiz, ordinate_horiz
        ax.invert_yaxis()
      plt.plot(abcissae_horiz, ordinate_horiz, '--', c='white')
      plt.plot(abcissae_verti, ordinate_verti, '--', c='white')
    
    return plt.gca()

  # -----------------------------------------------------------------------------

  def plot_3slices_new1(self, x, y, z, fig=None, contour=None, **kwargs):
    if fig is None:
      fig = figure(16,6)
    
    kwargs['vmin'] = kw('vmin', np.min(self), kwargs)
    kwargs['vmax'] = kw('vmax', np.max(self), kwargs)
    self.__log.debug('Setting vmin, vmax to: {}, {}'.format(kwargs['vmin'], 
                                                            kwargs['vmax']))
    
    # kwargs = dict(overwrite=0, overwrite_mmp=0, vmin=1500, vmax=7000, cmap='hsv')
    gs = fig.add_gridspec(2,2, height_ratios=[1,1], width_ratios=[2,1])
    fig.add_subplot(gs[0,0]) 
    ax = p.out.vp.it[it].plot(x=x, **kwargs)
    aspeqt(ax)
    fig.add_subplot(gs[1,0])
    ax = p.out.vp.it[it].plot(y=y, **kwargs)
    aspeqt(ax)
    fig.add_subplot(gs[:,1])
    ax = p.out.vp.it[it].plot(z=z, **kwargs)
    if contour is not None:
      colors = kw('colors', 'k', kwargs)
      levels = kw('levels', 40, kwargs)
      plt.contour(surf[...,0].T, extent=np.array(surf.extent[:-1]).flatten(), \
        colors=colors, levels=levels, alpha=0.4)
    ax.set_xlim(self.extent[ :2])
    ax.set_ylim(self.extent[2:4])
    aspeqt(ax)
    return ax

  # -----------------------------------------------------------------------------

  def plot_3slices(self, x, y, z, fig=None, gs=None, **kwargs):
    """
    """
    from fullwavepy.plot.plt2d import plot_image
    
    # layout = kw('layout', None, kwargs)
    # if layout is None:
      


    if fig is None:
      fig = figure(16,8)
    


    kwargs['x'] = x
    kwargs['y'] = y
    kwargs['z'] = z

    # LABELS FOR EACH AXIS
    s2 = kw('slice', 'y', kwargs) # MAIN SLICE PLOTTED AT THE BOTTOM IN FULL WIDTH
    s0, s1 = [i for i in ['x', 'y', 'z'] if i != s2]
    s = [s0, s1, s2]
    # CONVERT THE LABELS INTO ARRAY DIMENSIONS (AXES)
    convert_s2a = {'x': 0, 'y': 1, 'z': 2} # TRANSLATE slice TO axis
 
    if gs is None:
      gs = GridSpec(2,2)
   
    axes = list(np.zeros(3))
    axes[0] = fig.add_subplot(gs[0,0])
    axes[1] = fig.add_subplot(gs[0,1])
    axes[2] = fig.add_subplot(gs[1,:]) 
    
    kwargs['vmin'] = kw('vmin', np.min(self), kwargs)
    kwargs['vmax'] = kw('vmax', np.max(self), kwargs)
    self.__log.debug('Setting vmin, vmax to: {}, {}'.format(kwargs['vmin'], 
                                                            kwargs['vmax']))
    
    for i, ax in enumerate(axes):
      plt.sca(ax)
      aaxx = plot_image(np.take(self, kwargs[s[i]], convert_s2a[s[i]]), **kwargs)
      
      # PLOT SLICING LINES
      a, b = [j for j in ['x', 'y', 'z'] if j != s[i]]
      abcissae_horiz = range(self.shape[convert_s2a[a]])
      ordinate_horiz = np.full(len(abcissae_horiz), kwargs[b])
      ordinate_verti = range(self.shape[convert_s2a[b]])
      abcissae_verti = np.full(len(ordinate_verti), kwargs[a])
      
      if s[i] == 'z':
        abcissae_horiz, ordinate_horiz, abcissae_verti, ordinate_verti = abcissae_verti, ordinate_verti, abcissae_horiz, ordinate_horiz
        ax.invert_yaxis()
      plt.plot(abcissae_horiz, ordinate_horiz, '--', c='white')
      plt.plot(abcissae_verti, ordinate_verti, '--', c='white')
  
  # -----------------------------------------------------------------------------

  ###@widgets('cmap', 'slice', 'x', 'y', 'z')
  def plot_3slices_old1(self, fig=None, gs=None, widgets=False, **kwargs):
    """
    """
    from fullwavepy.plot.plt2d import plot_image
    
    if fig is None:
      fig = figure(16,8)
    
    kwargs['x'] = kw('x', 0, kwargs)
    kwargs['y'] = kw('y', 0, kwargs)
    kwargs['z'] = kw('z', 0, kwargs)

    # LABELS FOR EACH AXIS
    s2 = kw('slice', 'y', kwargs) # MAIN SLICE PLOTTED AT THE BOTTOM IN FULL WIDTH
    s0, s1 = [i for i in ['x', 'y', 'z'] if i != s2]
    s = [s0, s1, s2]
    # CONVERT THE LABELS INTO ARRAY DIMENSIONS (AXES)
    convert_s2a = {'x': 0, 'y': 1, 'z': 2} # TRANSLATE slice TO axis
 
    #if widgets: #FIXME BOILERPLATE
      #figsize = (kw('figsize_x', 8, kwargs), kw('figsize_y', 8, kwargs))
      #fig = plt.figure(figsize=figsize)
    
    if gs is None:
      gs = GridSpec(2,2)
      #gs = fig.add_gridspec(2,2)

    
    if widgets: #or fig is None:
      fig = figure(**kwargs)
      gs = fig.add_gridspec(2,2)
     

   
    axes = list(np.zeros(3))
    axes[0] = fig.add_subplot(gs[0,0])
    axes[1] = fig.add_subplot(gs[0,1])
    axes[2] = fig.add_subplot(gs[1,:]) 
    
    kwargs['vmin'] = kw('vmin', np.min(self), kwargs)
    kwargs['vmax'] = kw('vmax', np.max(self), kwargs)
    self.__log.debug('Setting vmin, vmax to: {}, {}'.format(kwargs['vmin'], 
                                                            kwargs['vmax']))
    kwargs['widgets'] = False
    self.__log.debug('Disabling widgets in inner functions.')
    
    
    for i, ax in enumerate(axes):
      plt.sca(ax)
      plot_image(np.take(self, kwargs[s[i]], convert_s2a[s[i]]), **kwargs)
      
      # PLOT SLICING LINES
      a, b = [j for j in ['x', 'y', 'z'] if j != s[i]]
      abcissae_horiz = range(self.shape[convert_s2a[a]])
      ordinate_horiz = np.full(len(abcissae_horiz), kwargs[b])
      ordinate_verti = range(self.shape[convert_s2a[b]])
      abcissae_verti = np.full(len(ordinate_verti), kwargs[a])
      
      if s[i] == 'z':
        abcissae_horiz, ordinate_horiz, abcissae_verti, ordinate_verti = abcissae_verti, ordinate_verti, abcissae_horiz, ordinate_horiz
        ax.invert_yaxis()
      plt.plot(abcissae_horiz, ordinate_horiz, '--', c='white')
      plt.plot(abcissae_verti, ordinate_verti, '--', c='white')
    
    #return ax1, ax2, ax3
    
  # -----------------------------------------------------------------------------  
    
  def plot(self, *args, **kwargs):
    """
    Framework plotter.
    
    Notes
    -----
    This is a preferred function to call rather than
    plot_3slices directly. This is because plot 
    formatting is set in subclasses by overwriting
    plot method. This could be avoided by defining
    _format_plot() method or similar.

    Note, it doesn't need to have ##@widgets!
    
    """
    if not ('x' in kwargs or 'y' in kwargs or 'z' in kwargs):
      nslices = 1
    elif 'x' in kwargs and not ('y' in kwargs or 'z' in kwargs):
      nslices = 1
      kwargs['slice_at'] = 'x'
      kwargs['node'] = kwargs['x']
    elif 'y' in kwargs and not ('x' in kwargs or 'z' in kwargs):
      nslices = 1
      kwargs['slice_at'] = 'y'
      kwargs['node'] = kwargs['y']
    elif 'z' in kwargs and not ('x' in kwargs or 'y' in kwargs):
      nslices = 1
      kwargs['slice_at'] = 'z'
      kwargs['node'] = kwargs['z']  
    elif 'x' in kwargs and 'y' in kwargs and 'z' in kwargs:
      nslices = 3
    else:
      raise ValueError('Slicing arguments not understood.')
    
    if nslices == 1:
      self.plot_slice(*args, **kwargs)
    elif nslices == 3:
      self.plot_3slices(*args, **kwargs)
    else:
      raise ValueError('Wrong value of nslices: %s' %str(nslices))
    
    return plt.gca()

  # -----------------------------------------------------------------------------
  
  def scroll(self, **kwargs):
    """
    
    """
    import matplotlib.pyplot as plt
    from fullwavepy.plot.events import IndexTracker
    
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, self, **kwargs)
    return fig, ax, tracker

  def scrollall(self, fig, **kwargs):
    """
    To make it work in a jupyter notebook:
     %matplotlib notebook
     %matplotlib notebook
     fig = plt.figure(figsize=(5,20))
     tracker = some_array.scrollall(fig, cmap='viridis')
     fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    
    """
    from fullwavepy.plot.events import IndexTrackerAll
    
    tracker = IndexTrackerAll(fig, self, **kwargs)
    return tracker
    #return tracker.onscroll

  # -----------------------------------------------------------------------------
@traced
@logged
class GenericPoint(np.ndarray):
  """
  We want to have lists of the instances of this class
  identified with a unique ID (optional).
  We don't want dicts as we will often want to keep them ordered.

  """
  def __new__(cls, xyz, value=1, **kwargs):
    obj = np.asarray(xyz).view(cls)
    obj.value = value
    obj.ID = kw('ID', None, kwargs)
    return obj

  # -----------------------------------------------------------------------------
  
  def __array_finalize__(self, obj):
    if obj is None: return

  # -----------------------------------------------------------------------------

  def find_neighs(self, r, **kwargs):
    """
    Find a cube of nodes surrounding the point.
    
    Notes
    -----
    Works in any dimension (checked 1-3), although in 1D 
    the resultant array should be flattened.
    
    Last axis of np.array(np.meshgrid(*ranges, indexing='ij')).T
    is a tuple of coordinates of a given point. e.g. (x,y,z) in 3D.
    
    We swap the first and the second (!) but last axis to have the Z-coordinate 
    to be the last and thus fastest-changing (np.arrays are row-major) index.
    
    """
    from fullwavepy.numeric.generic import neighs1d

    ranges = []
    for i in self:
      ranges.append(neighs1d(i, r))
    
    self.neighs = np.array(np.meshgrid(*ranges, indexing='ij')).T.swapaxes(0,-2)
    return self.neighs

  # -----------------------------------------------------------------------------    
  
  def spread(self, r, funcs, **kwargs):
    """
    Spread the point onto a cuboid using
    in general different functions along each coordinate 
    axis.
    
    Parameters
    ----------
    funcs : list
    
    Returns
    vol : Arr3d
      3D array of values.
      It is endowed with vol.coords and vol.extent attributes.
      The former stores coordinates.

    Notes
    -----
    See Hicks 2002, Geophysics for details.
    
    We use the same r for neighbours and the window as 
    outside the window values are zero by definition.
    
    """
    assert len(funcs) == len(self)
    
    # CUBE OF (x,y,z) TUPLES
    cube = self.find_neighs(r, **kwargs)
    # CENTER THE COORDINATE SYSTEM AT self
    dists = cube - self
    # ND-DELTA IS A PRODUCT (!) OF 1D-ONES
    vol = np.ones(dists[...,0].shape)
    # APPLY ALONG TUPLE AXIS, I.E. TAKE POINTS COORDS AS AN ARGUMENT
    coord_axis = -1
    # DEAL WITH ONE COORDINATE AT A TIME
    extent = [] # WHAT IS THIS FOR?! FIXME
    for i, func in enumerate(funcs):
      # WRAPPER TO ACT ON A SINGLE COORDINATE OF AN ND-POINT
      func_of_xyz = lambda point : func(point[i])
      vol *= np.apply_along_axis(func_of_xyz, coord_axis, dists)
    
    
    #nshape = np.array(cube.shape)
    #nshape[-1] += 1 # INCREASE THE LAST DIM TO INCLUDE VALUE -> (x,y,z,val)
    #
    #self.vol = np.zeros(nshape)
    #self.vol[..., :-1] = cube
    #self.vol[..., -1] = vol
    
    self.__log.debug('Scaling the cube with self.value=%s' % str(self.value))
    self.vol = Arr3d(vol) * self.value
    self.vol.extent = self.cube_extent(cube, **kwargs)
    #self.vol._set_coords()
    self.__log.debug('self.vol.extent' + str(self.vol.extent))
    self.vol.coords = cube
    return self.vol

  # -----------------------------------------------------------------------------
  
  def cube_extent(self, cube, **kwargs):
    if len(cube.shape) == 4:
      x1, y1, z1 = cube[0,0,0]
      self.__log.debug('Modified  extent of cube - tmp? DOUBLE-CHECK')
      x2, y2, z2 = cube[-1,-1,-1] # + 1
      extent = [[x1, x2], [y1, y2], [z1, z2]]

    else:
      raise ValueError('cube.shape ' + str(cube.shape))
    
    self.__log.debug('returning extent %s' % str(extent))    
    return extent
  
  # -----------------------------------------------------------------------------
@traced
@logged
class Points(object):
  """
  Apparently subclassing a list is not the best idea,
  see 'composition over inheritance'.
  
  https://stackoverflow.com/questions/25328448/should-i-subclass-python-list-or-create-class-with-list-as-attribute
  (...) whenever you ask yourself "should I inherit or have a member of that type", 
  choose not to inherit.
  This rule of thumb is known as "favour composition over inheritance". 
  
  Another reason is that __new__ that we have to overwrite is not loggable.

  Obsolete too:
  Changed from dict to list.
  DON'T KNOW WHY list.__new__ returns None

  Obsolete:
  dict stores ids (keys) and coords (values)
  ids can annotate plots
  
  __new__ is not necessary to redefine, unlike nd.array????
  
  """  
  # def __new__(cls, source, **kwargs):
  #   cls.__log.debug('source %s' % str(source))
  #   to_return = super().__new__(cls, source, **kwargs)
  #   cls.__log.debug('to_return %s' % str(to_return))
  #   return to_return
  def __init__(self, li, **kwargs):
    """
    li : list
    
    """
    self.li = li

  # -----------------------------------------------------------------------------
@traced
@logged
class Points3d(Points):
  """
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    for elem in self.li: # checking if it's really (x,y,z)
      assert np.array(elem).shape == (3,) 

  # -----------------------------------------------------------------------------

  def slice(self, slice_at='y', **kwargs):
    if slice_at == 'x':
      i1, i2 = 1, 2
    elif slice_at == 'y':
      i1, i2 = 0, 2
    elif slice_at == 'z':
      i1, i2 = 0, 1
    else:
      raise ValueError('Wrong slice coord: %s' % slice_at)      
    for i, elem in enumerate(self.li):
      #assert len(val) == 3 # IT CAN HAVE METADATA
      self.li[i] = np.array([elem[i1], elem[i2]])

  # ----------------------------------------------------------------------------- 

  def plot_slice(self, ax=None, **kwargs):
    """
    """
    annotate = kw('annotate', False, kwargs)
    annoffset = kw('annoffset', 0, kwargs)
    alpha = kw('alpha', 0.7, kwargs)
    marker = kw('marker', '.', kwargs)
    markersize = kw('markersize', 5, kwargs)
    markeredgecolor = kw('markeredgecolor', 'k', kwargs)
    markerfacecolor = kw('markerfacecolor', 'none', kwargs) # EMPTY MARKERS
    if ax is None:
      ax = plt.gca()
    
    self.slice(**kwargs)
    
    if annotate: 
      for key, val in self.items():
        ax.annotate(key, (val[0]+annoffset, val[1]+annoffset), clip_on=True) # clip_on IS REQUIRED
    
    ax.plot([i[0] for i in self.li], [i[1] for i in self.li], 
            '.',
            alpha=alpha, 
            marker=marker, 
            markersize=markersize, 
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
           )
  
  # -----------------------------------------------------------------------------   
  
  def plot_3slices(self, fig, **kwargs): # LEGACY
    d = self.read(**dict(kwargs, unit='node'))
    
    s3 = kw('slice', 'y', kwargs) #FIXME: THIS MUST BE MERGED WITH arr3d
    s1, s2 = [i for i in ['x', 'y', 'z'] if i != s3]
    s = [s1, s2, s3]
    
    for i in range(3):
      self.plot_slice(s[i], fig.axes[i])

  # -----------------------------------------------------------------------------   

  def plot(self, *args, **kwargs):
    #if 'slice_at' in kwargs:
    self.plot_slice(*args, **kwargs)
    #else:
     
  # -----------------------------------------------------------------------------
  
  def plotly(self, fig=None, **kwargs): # LEGACY
    """
    """
    import plotly.graph_objects as go    
    color = kw('color', 'black', kwargs)
    mode = kw('mode', 'markers', kwargs)
    size = kw('size', 2, kwargs)
    
    if fig is None:
      fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i[0] for i in self.li], 
                             y=[i[1] for i in self.li], 
                             text=[i.ID for i in self.li], mode=mode,
                             marker=dict(color=color, size=size),
                             line=dict(color=color), showlegend=False))
    
    return fig       
     
  # -----------------------------------------------------------------------------   
@traced
@logged
class SRs(Points3d):
  """
  li : list
    List of [id, xyz] records, where xyz = (x,y,z)
    
  """
  def __init__(self, li, **kwargs):
    self.__log.debug('li: %s' % str(li))
    self.li = []
    for l in li:
      ID, xyz = l
      self.li.append(PointSR(xyz, ID=ID))
    
    # cls.__log.debug('type(new_li[0]) %s' % type(new_li[0]))
    # print(new_li)
    # return super().__new__(cls, new_li)
    # return new_li
    # super().__init__(new_li, **kwargs)

  # -----------------------------------------------------------------------------

  def set_type(self, srtype_ids, **kwargs):
    """
    it will be read from the file pgy / geo instead...?
    
    PROTEUS convention of naming data components.
    
    """
    mapp = {0 : Monopole,
            1 : DipoleZ, # NOT X!
            2 : DipoleY,
            3 : DipoleX,
           }
    
    assert len(srtype_ids) == len(self.li)
    # for srtype_id, sr[k, v] in zip(srtype_ids, self.items()):
    new_list = []
    for srtype_id, sr in zip(srtype_ids, self.li):
      clss = mapp[srtype_id]
      self.__log.debug('Appending instance of %s' % str(clss))
      new_list.append(clss(sr, ID=sr.ID, **kwargs))

    self.__log.debug('new_list: %s' % str(new_list))
    self.li = new_list
  
  # ----------------------------------------------------------------------------- 
 
  def spread_factors(self, srtype_ids, *args, **kwargs):
    """
    """
    self.set_type(srtype_ids, **kwargs)
    #self.sprd_fctrs = {}
    self.hyper = {}
    nsr = len(self.li)
    i = 1
    # for srid, sr in self.items():
    for sr in self.li:
      srid = sr.ID
      self.__log.info('ID %s (%s/%s)' % (srid, i, nsr))
      self.hyper[srid] = HyperPointSR(sr, ID=srid, **kwargs)
      self.hyper[srid].spread_factors(*args, **kwargs)
      i += 1
  
  # -----------------------------------------------------------------------------
  
  def qc_sprd_fctrs(self, **kwargs):
    pass
  
  def widgets_qc_sprd_fctrs(self, **kwargs):
    snap_max = 1
    raise ValueError(snap_max)
    widgets = dict(srid=Dropdown(options=[i.ID for i in self.li]),
                   snap=IntSlider(value=snap_max, min=0, max=snap_max, step=1),
                   slice_at=Dropdown(options=['y', 'x', 'z']),
                   node=IntSlider(value=rmax, min=0, max=2*rmax-1, step=1))
    return widgets
@traced
@logged
class Sources(SRs):
  def plot(self, *args, **kwargs):
    kwargs['marker'] = kw('marker', '*', kwargs)
    kwargs['markersize'] = kw('markersize', 10, kwargs)
    kwargs['markeredgecolor'] = kw('markeredgecolor', 'k', kwargs)
    kwargs['markerfacecolor'] = kw('markerfacecolor', 'w', kwargs)
    super().plot(*args, **kwargs)

  # -----------------------------------------------------------------------------

  def plotly(self, *args, **kwargs):
    kwargs['mode'] = kw('mode', 'markers', kwargs)
    kwargs['color'] = kw('color', 'black', kwargs)
    kwargs['size'] = kw('size', 2, kwargs)
    return super().plotly(*args, **kwargs)
@traced
@logged
class Receivers(SRs):
  def plot(self, **kwargs):
    kwargs['annotate'] = False
    kwargs['s'] = 1e-2
    kwargs['c'] = 'gray'
    kwargs['alpha'] = 1
    super().plot(**kwargs)
  
  # -----------------------------------------------------------------------------  

  def plotly(self, *args, **kwargs):
    kwargs['mode'] = kw('mode', 'markers', kwargs)
    kwargs['color'] = kw('color', 'grey', kwargs)
    kwargs['size'] = kw('size', 1, kwargs)
    return super().plotly(*args, **kwargs)

  # ----------------------------------------------------------------------------- 
@traced
@logged
class HyperPointSR(object):
  """
  FIXME: implement check if we're not out of bounds
  
  """
  def __init__(self, pointsr, **kwargs):
    """
    """
    self.pointsr = pointsr
    self.r_hicks = kw('r_hicks', 3, kwargs)

  # -----------------------------------------------------------------------------
  
  @timer
  def find_vol(self, **kwargs):
    """
    """
    rmax = kw('rmax', 10, kwargs)
    
    coords = self.pointsr.find_neighs(rmax, **kwargs)
    self.origin = coords[0,0,0] # COORDINATES OF THE FIRST NODE 
    
    
    # THIS IS THE CUBOIDAL VOLUME OF SPREAD FACTORS 
    # IT WILL BE USED TO INJECT THE SOURCE INTO (OR INTERPOLATE RECEIVER)
    # THE WAVEVIELD
    sh = np.array(coords.shape)
    sh[-1] += 1
    self.vol = np.zeros(sh)
    self.vol[..., :3] = coords
    self.vol[..., 3] = 0.0 # this will store amplitude, i.e. sprd_fctrs
    
  # -----------------------------------------------------------------------------
  
  
  # FIXME: should not depend on ghs, etc. in some cases
  @timer
  def spread_factors(self, ghs, iss, ine, elef, efro, etop, **kwargs):
    """
    Iterative 
    
    Notes
    -----
    We pass kwargs to allow for 

    """
    assert isinstance(ghs, np.ndarray)

    self.find_vol(**kwargs)
    self.vol[..., -1] = 0.0 # OTHERWISE GROWING IN IPYNB
    
    psrs_to_spread = [self.pointsr]
    i_max = 2
    i = 0
    self.snapshots = []
    while i <= i_max:
      n_to_spread = len(psrs_to_spread)
      self.__log.info('It. %s/%s: %s points to spread...' % (i, i_max, n_to_spread)) 

      if n_to_spread == 0:
        self.__log.info('No more points to spread. Returning.')
        return
      
      self.__log.debug('psrs_to_spread' + str(psrs_to_spread))
      new_psrs_to_spread = []
      for psr_to_spread in psrs_to_spread:
        self.__log.debug('psr_to_spread ' + str(psr_to_spread))
        
        del_kw('r_hicks', kwargs)
        psr_to_spread.spread_factors(r_hicks=self.r_hicks, **kwargs)
        psr_to_spread.vol.split(ine, elef, efro, etop, **kwargs)
        
        self._update_vol(np.copy(psr_to_spread.vol.ins), **kwargs)
        
        
        reflected = psr_to_spread.vol.bounce_off(ghs, iss, elef, efro, etop, **kwargs)
        if len(reflected) > 0:
          new_psrs_to_spread += reflected # append would create a nested list (undesired)
      
      self.snapshots.append(np.copy(self.vol))
      psrs_to_spread = list(new_psrs_to_spread)
      i += 1

  # -----------------------------------------------------------------------------
  
  @timer
  def _update_vol(self, nodes_in, **kwargs):
    for node_in in nodes_in:
      #self.__log.info('node_in ' + str(node_in))
      xyz = node_in[:3]
      amp = node_in[3]
      if abs(amp) < clip_at: # skip tiny factors for speed and compactness of volume
        self.__log.debug('Skipping inside node %s with small ampl %s' % (str(xyz), str(amp)))
        continue      
      
      ijk = xyz - self.origin
      ijk = tuple((int(n) for n in ijk))
      try:
        prv = self.vol[ijk][-1]
      except IndexError:
        self.__log.debug('Skipping inside (medium) node %s which is outside the volume' % (str(xyz)))
        continue
      
      
      now = prv + amp
      self.vol[ijk][-1] = now
      #if abs(amp) > 1e-4:
      if True:
        self.__log.debug('Updated %s by %s from %s to %s ' % (str(ijk), 
                                                             str(amp), 
                                                             str(prv),
                                                             str(now)))

  # -----------------------------------------------------------------------------
@traced
@logged
class PointSR(GenericPoint):
  """
  ATTENTION 
  It is supposed to have coordinates in nodes 
  of the regular (not extended!) grid.
  NOTE
  Left-front-top corner of the grid is (1,1,1)
  Right-back-bottom corner of the grid is (nx1,nx2,nx3)
  
  Notes
  -----
  Is it really necessary?
  Monopole etc. could just inherit from GenericPoint directly...
  It is, provided VolumeSR is attribute of PointSR, not GenericPoint
  (even if we rename it to GenericVolume?). And we DO NEED VolumeSR 
  which has unique methods such as split and bounce_off.
  
  """
  @timer
  def spread_factors(self, r_hicks, **kwargs):
    """
    This wrapper-class probably!  can't be avoided as it transmits
    info between children and parent. (needs different name than spread)
    """
    self.vol = self.spread(r_hicks)

  # -----------------------------------------------------------------------------
  
  @timer
  def spread(self, *args, **kwargs):
    """
    args are just passed from children to parent.
    this is especially useful here since the funcs
    used by the parent
    are defined in children under the hood.
    that's why we couldn't spread an instance
    of PointSR without providing them.
    
    """
    vol = super().spread(*args, **kwargs)
    self.__log.debug('vol.extent' + str(vol.extent))

    self.vol = VolumeSR(vol)
    self.vol.extent = vol.extent
    self.vol.coords = vol.coords
    return self.vol

  # -----------------------------------------------------------------------------    
@traced
@logged
class VolumeSR(Arr3d):
  """
  """
  @timer
  def split(self, *args, **kwargs):
    """
    """
    self.flgs = self.flag(*args, **kwargs)
    
    attrs = ['ins', 'at', 'out']
    flags = [self.in_flag, self.acc_flag, self.ext_flag]

    for attr, flag in zip(attrs, flags):
      indices = self.flgs == flag
      coords = self.coords[indices]
      values = self[indices]
      arr = np.zeros(list(values.shape) + [4]) # must be neater syntax
      arr[..., 0:-1] = coords
      arr[..., -1] = values
      self.__log.debug('Found %s %s-nodes' % (len(arr), attr))
      setattr(self, attr, arr)

  # -----------------------------------------------------------------------------    
  
  @timer
  def flag(self, ine, *args, **kwargs):
    """
    ine: array of interior/accurately at FS/exterior nodes
    """
    self.indices = self._grid_coords_2_egrid_indices(*args, **kwargs)
    
    self.flgs = Arr3d(ine[self.indices])
    self.flgs.extent = self.extent
    
    self.in_flag = ine.in_flag
    self.acc_flag = ine.acc_flag
    self.ext_flag = ine.ext_flag
    
    return self.flgs
  
  # -----------------------------------------------------------------------------
  
  @timer
  def _grid_coords_2_egrid_indices(self, elef, efro, etop, **kwargs):
    """
    Rationale
    ---------
    use vol.coords to pick subarray from inext-nodes array etc
    
    convert coords (x,y,z) of grid:
    x = 1, 2, ... nx1 
     
    (coords is an array as vol.coords laid out with 
    x  assumed to be the slowest and z the fastest index)
     
    to indices (!) of the array storing whole extended (!) grid 
     i = 0, ..., enx1
     ...
    order of indices as in coords  
     
    """
    grid = np.copy(self.coords.T.swapaxes(1,-1))
    grid[0] += elef
    grid[1] += efro
    grid[2] += etop
    indices = np.copy(grid-1) # subtract  1 to convert array indices!!!!!!
    indices = tuple(indices) # CRUCIAL, OTHERWISE SLICING WOULDN'T WORK
    return indices

  # -----------------------------------------------------------------------------
  
  @timer
  def bounce_off(self, ghs, iss, elef, efro, etop, **kwargs):
    """
    ghs : list of ghosts
    iss : list of corresponding intersects (same len)
    
    ATTENTION 
    all the calculation here is done in extended-grid coordinates.
    But PointSR.spread assumes regular (not extended) grid, so we 
    need to convert it back.
    
    """
    assert isinstance(ghs, np.ndarray) # lists are terribly slow and np.array(list)
    # for each point is even worse an idea. This was a 10x bottle-neck
    assert len(ghs) == len(iss)
    
    # t1 = time.perf_counter()
    vout = np.copy(self.out)
    # t2 = time.perf_counter()
    # self.__log.info('Copying self.out to vout took %s s' % "{:15.12f}".format(t2-t1))
    
    # t1 = time.perf_counter()
    vout[:, 0] += elef
    vout[:, 1] += efro
    vout[:, 2] += etop
    # t2 = time.perf_counter()
    # self.__log.info('Adding extra nodes to vout took %s s' % "{:15.12f}".format(t2-t1))    
    # here we compare grid-coords, NOT grid-coords to array-indices
    # => we don't subtract 1
    

    # THIS A BOTTLENECK!!!
    # t1 = time.perf_counter()
    # aghs = np.array(ghs)
    # t2 = time.perf_counter()
    # self.__log.info('aghs = np.array(ghs) took %s s' % "{:15.12f}".format(t2-t1))    

    reflected = []
    for vo in vout:
      x, y, z, amp = vo
      if abs(amp) < clip_at: # skip tiny factors for speed and compactness of volume
        self.__log.debug('Skipping outside node %s with small ampl %s' % (str([x,y,z]), str(amp)))
        continue
      
      # t1 = time.perf_counter()
      Gs = ghs[((ghs[:,0] == x) & (ghs[:,1] == y) & (ghs[:,2] == z))]
      # t2 = time.perf_counter()
      # self.__log.info('Locating the ghost took %s s' % "{:15.12f}".format(t2-t1))       
      
      if len(Gs) == 1:
        G = Gs[0]
      elif len(Gs) > 1:
        raise ValueError('More than one ghost found: ' + str(Gs))
      else:
        self.__log.warning('No ghost found for the outside node %s. Skipping it.' % str([x,y,z]))
        continue
      
      I = np.array(iss[np.where(np.all(ghs==G, axis=1))[0][0]])
      # I = np.array(iss[ghs.index(list(G))]) # this was the bottle-neck (see above)
      G = G[:3]
      
      #R[:3] = 2 * I - G
      #R[3] = -amp
      

      R = np.zeros(3) # WE HAVE TO CREATE IT EVERY TIME
      R = 2 * I - G # FIND A REFLECTION OF G WITH RESPECT TO I

      # NOTE return to not-extended-grid coordinates
      R[0] -= elef
      R[1] -= efro
      R[2] -= etop
      
      # NOTE my theory is we should always inject secondary (reflected) sources  
      # as monopoles regardless of the type of the primary source
      R = Monopole(R)
      R.value = -amp # flip the polarity as we reflect from a free surface (it will change for other boundaries!)
      
      reflected.append(R)
      #print('O', vo)
      #print('G', G)
      #print('I', I)
      #print('R', R)
      #print()   
    return reflected #np.array(reflected)

  # -----------------------------------------------------------------------------
  
  def plot(self, **kwargs):
    kwargs['slice_at'] = kw('slice_at', 'y', kwargs)
    kwargs['node'] = kw('node', self.shape[1]//2, kwargs)
    super().plot(**kwargs)

  # -----------------------------------------------------------------------------
@traced
@logged
class Monopole(PointSR):
  """
  """
  def spread(self, r, **kwargs):
    func = lambda x : kaiser(x, r) * sinc(x)
    funcs = [func for i in range(len(self))]
    return super().spread(r, funcs, **kwargs)
@traced
@logged
class Dipole(PointSR):
  """
  axis : 0, 1 or 2
    corresponds to dipole along X, Y or Z axis respectively
  
  """
  def __new__(cls, xyz, axis, **kwargs):
    assert axis in [0, 1, 2]
    cls.axis = axis
    return super().__new__(cls, xyz, **kwargs)
  
  # -----------------------------------------------------------------------------
  
  def spread(self, r, **kwargs):
    """
    Notes
    -----
    Particle velocity is proportional 
    to the NEGATIVE pressure gradient (v_i ~ -dp/dx_i),
    hence the minus sign in func2.

    """
    func1 = lambda x : kaiser(x, r) * sinc(x) 
    func2 = lambda x : kaiser(x, r, dipole=True) * dsinc_dx(x) * (-1)
    
    funcs = [func1 for i in range(len(self))]
    funcs[self.axis] = func2
    
    return super().spread(r, funcs, **kwargs)

  # -----------------------------------------------------------------------------
@traced
@logged
class DipoleX(Dipole):
  def __new__(cls, xyz, **kwargs):
    return super().__new__(cls, xyz, 0, **kwargs)  
@traced
@logged
class DipoleY(Dipole):
  def __new__(cls, xyz, **kwargs):
    return super().__new__(cls, xyz, 1, **kwargs) 
@traced
@logged
class DipoleZ(Dipole):
  def __new__(cls, xyz, **kwargs):
    return super().__new__(cls, xyz, 2, **kwargs) 
@traced
@logged
def xyz2w(xyz, extended_dims, **kwargs):
  """
  """
  x, y, z = xyz
  nx, ny, nz = extended_dims
  return int((x - 1) * ny * nz + (y - 1) * nz + z)
@traced
@logged
def w2xyz(w, extended_dims, **kwargs):
  """
  For QC.
  
  FIXME: it's terribly slow so now only for 
  single-value checks. We could use Adrian's 
  formulas to vectorize it.
  
  """
  assert float(w).is_integer()
  assert w > 0
  enx, eny, enz = extended_dims
  
  if w > enx * eny * enz:
    raise ValueError('w > enx * eny * enz')


  i = 1
  for x in range(1, enx+1):
    for y in range(1, eny+1):
      for z in range(1, enz+1):
        if i == w:
          return x, y, z
        i += 1
  raise ValueError('Could not find x,y,z for w=%s' % w)

