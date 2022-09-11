'''io utilities for remote sensing multi/hyperspectral imagery'''

## these are kind of hodge-podge right now...


import h5py
import numpy as np
import verbose as v
try:
    import spectral
    SPY_AVAIL=True
except ImportError:
    SPY_AVAIL=False

if SPY_AVAIL:

    def write_envi_image(filename,img):
        '''write ENVI formatted file with img data in it'''
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0],img.shape[1],1)
        v.vprint("Writing to",filename,"and",filename+'.hdr')
        spectral.envi.save_image(filename+'.hdr',img,interleave='bip')


## read_rsim (read remote sensing image, multispectral image cube, read_cube?
## where, like read_seq, it guesses type of image based on file extension
## 2d data: array, matrix, grid, bag-o-pixels,

def get_file_tag(filename,tag=None):
    '''use @ in filename = file@tag; return file,tag'''
    try:
        hdf_file,xtag = filename.split("@",1)
    except ValueError:
        hdf_file = filename
        xtag = None
    tag = xtag or tag or 'cube'
    return hdf_file,tag

def write_cube(filename,cube,tag=None,interleave='BIP'):
    '''write image cube to hdf file'''
    ## cube will be written as-is, so interleave should
    ## specify the interleave appropriate to the cube
    hdf_file,tag = get_file_tag(filename,tag=tag)
    with h5py.File(hdf_file,'w') as f:
        dset = f.create_dataset(tag,data=cube)
        dset.attrs.create('interleave',interleave)

def read_cube(filename,tag=None,interleave=None,bandsample=None):
    '''
    read hdf5 file; return image cube
    '''
    ## Need to decide is 'interleave' telling you what the input interleave is
    ## Or is it telling you what output interleave is desired
    ## Add some code to enable filename@tag as input
    hdf_file,tag = get_file_tag(filename,tag=tag)
    with h5py.File(hdf_file,'r') as f:
        v.vprint('f:',f.keys())
        if tag not in f.keys():
            raise RuntimeError(f'tag={tag} not available, '
                               f'consider: {f.keys()}')
        cube = f[tag]
        v.vprint("init shape:",cube.shape)

        cube_interleave = cube.attrs.get('interleave',None)
        if interleave and cube_interleave and cube_interleave != interleave:
            raise RuntimeError("Confliciting interleave")
        interleave = interleave or cube_interleave

        v.vprint("interleave=",interleave)
        cube = np.array(cube).astype(float)
        if interleave == 'BSQ':
            cube = np.moveaxis(cube,0,-1)
            interleave = 'BIP'

        if bandsample:
            cube = cube[:,:,::bandsample]

        return cube

def read_cube_target(filename,
                     tag='cube',
                     targettag='target',
                     interleave=None,
                     bandsample=None):
    '''
    read hdf5 file; return image cube, target spectrum,
    and (optionally) array of associated wavelenghts
    '''
    with h5py.File(filename,'r') as f:
        cube = f[tag]
        v.vprint("init shape:",cube.shape)

        cube_interleave = cube.attrs.get('interleave',None)
        if interleave and cube_interleave and cube_interleave != interleave:
            raise RuntimeError("Confliciting interleave")
        interleave = interleave or cube_interleave

        v.vprint("interleave=",interleave)
        cube = np.array(cube).astype(float)
        if interleave == 'BSQ':
            cube = np.moveaxis(cube,0,-1)
            interleave = 'BIP'

        v.vprint('shape:',cube.shape,np.min(cube),np.max(cube))

        ## Read target spectrum
        if targettag:
            tgt = f[targettag]
            tgt = np.array(tgt).reshape(-1)
        else:
            tgt = None

        try:
            w = f['lambda']
            w = np.array(w).reshape(-1)
        except KeyError:
            w = None

    if bandsample:
        cube = cube[:,:,::bandsample]
        tgt = tgt[::bandsample]
        if w is not None:
            w = w[::bandsample]

    return cube,tgt,w
