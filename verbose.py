'''verbose: print statements that depend on a user-set level of verbosity'''

import sys
import builtins
import functools
from collections import Counter

import tqdm as tqdmx ## allows to use 'tqdm' as component of verbose
def _tqdm(iter,**kw):
    '''wrapper for tqdm'''
    return tqdmx.tqdm(iter,**kw)
def _tqdm_null(iter,**kw):
    '''just pass the iterator through with no tqdm progress bar'''
    return iter

V_MAX_LEVELS=5

class VerboseClass:
    '''
    encapsulates vprint functions
    that write messages to stderr
    based on user-set level of verbosity

    Typical use:
       from verbose import verbose as v
       ...
       v.verbosity(1)
       ...
       v.vprint("read",n_sequences,"seqeunces from file")
       ...
    As an alternative to:
       verbosity_level=1
       ...
       if verbosity_level > 0:
           print("read",n_sequences,"sequences from file",file=sys.stderr)
       ...

    Functions provided:
       v.print(...)  ==  print(...,file=sys.stderr)
       v.vprint(...) ==  if verbosity_level > 0: print(...,file=sys.stderr)
       Also: v.vvprint(...), v.vvvprint(...) for higher verbosity levels

       v.print_only(maxcount,label,...)
       v.vprint_only(maxcount,label,...)
       v.vvprint_only(maxcount,label,...)
         prints message at most maxcount times, then goes silent

       v.print_only_summary(label,...)
         prints a summary saying how often print_only /would/ have printed
         if only it weren't limited by maxcount

    Special functions:
       v.tqdm(...) == tqdm.tqdm
       v.vtqdm(...) == tqdm.tqdm if verbose else lambda x: x

    '''

    ## class variables
    verbose_level=0
    initialized=False
    count=Counter() ## currently no way to reset the counter, any need to?

    @classmethod
    def initialize(cls):
        '''create classmethods vprint, vvprint, vvvprint, etc'''
        if cls.initialized:
            ## in case class has already been initialized
            ## don't need to re-initialize ... even though
            ## re-initializing doesn't appear to hurt anything
            return

        ## make list of available functions
        cls.vfunctions = {
            'print': cls.vnprint,
            'print_only': cls.vnprint_only,
            'print_only_summary': cls.vnprint_only_summary,
            'tqdm': cls.vntqdm
        }

        ## make all those functions
        for vlevel in range(V_MAX_LEVELS):
            for fcn_name,fcn in cls.vfunctions.items():
                cls.mk_classfcn(fcn,fcn_name,vlevel)
        cls.initialized=True

    @classmethod
    def mk_classfcn(cls,vn_fcn,fcn_name,vlevel):
        '''make a classmethod such as vprint, vvprint, etc'''
        setattr(cls,'v'*vlevel + fcn_name,
                functools.partial(vn_fcn,vlevel))

    @classmethod
    def verbosity(cls,vlevel=None):
        '''user sets the level of verbosity; 
        call with no arg to query verbosity'''
        if vlevel is not None:
            cls.verbose_level = vlevel
        if not cls.initialized:
            cls.initialize()
        return cls.verbose_level

    @classmethod
    def vntqdm(cls,vlevel,iter,**kw):
        '''
        if verboseity leven is vlevel or higher,
        then wrap iterator in tqdm for progress bar
        '''
        if cls.verbose_level >= vlevel:
            return _tqdm(iter,**kw)
        else:
            return _tqdm_null(iter)

    @classmethod
    def vnprint(cls,vlevel,*p,**kw):
        '''
        if verbosity level is vlevel or higher, then print
        to sys.stderr and flush every print
        '''
        if cls.verbose_level >= vlevel:
            builtins.print(*p,file=sys.stderr,flush=True,**kw)

    ## Variant of vprint that only prints the warning a fixed
    ## maximum number of times; thus warning you that something
    ## is happening, but not scrolling pages and pages of it by
    ## your screen.  Later, ie outside the loop you can (optionally)
    ## call for a summary of how many times the warning was triggered.

    ## common usage might be:
    ## from verbose import verbose as v
    ## ...
    ## for data_sample in data:
    ##    if is_bad(data_sample):
    ##         v.vprint_only(5,'Bad data',data_sample.info)
    ##    ...
    ## v.vprint_only_summary('Bad data')

    ## Note that the warning string is a key.  That means you
    ## can have several of these going at the saime time, identifying
    ## different potential causes for a warning.  But it also means
    ## that you cannot put changing information into that key string

    ## Okay:
    ## for data_sample in data:
    ##     if is_bad(data_sample):
    ##         v.vprint_only(5,'Bad data',data_sample.info)
    ##     if too_big(data_sample):
    ##         v.vprint_only(5,'Too big','Data sample has',data_sample.size,'units'))
    ##    ...
    ## v.vprint_only_summary('Bad data')
    ## v.vprint_only_summary('Too big')

    ## Mistake:
    ##    vprint_only(5,f'Bad data: {data_sample.info}')
    ##

    @classmethod
    def vnprint_only_summary(cls,vlevel,msg,*p,**kw):
        '''
        summarize how many times vprint_only was called
        with msg as first arguemnt; keep quiet if count was zero
        '''
        if cls.count[msg]:
            cls.vnprint(vlevel,msg,cls.count[msg],*p,**kw)

    @classmethod
    def vnprint_only(cls,vlevel,maxcount,msg,*p,**kw):
        '''vprint, but only maxcount times, at most'''
        cls.count[msg] += 1
        if cls.count[msg] <= maxcount:
            cls.vnprint(vlevel,msg,*p,*kw)

## Need to initialize or else you'll get mysterious AttributeError
## If your application calls v.vprint, say, before calling v.verbosity(...)
VerboseClass.initialize()

if 1:
    ## Kind of a kluge, but it enables verbose
    ## to be treated as a module; thus: 'import verbose as v'

    verbosity = VerboseClass.verbosity

if 1:
    ## Even more of a kluge, and uses 'exec' function
    for _vlevel in range(V_MAX_LEVELS):
        vn = 'v'*_vlevel
        for fname in VerboseClass.vfunctions:
            exec(f'{vn}{fname} = VerboseClass.{vn}{fname}')

if 0:
    ## Here's what that exec was doing,
    ## This avoids exec, but hard to change V_MAX_LEVELS

    print = VerboseClass.print
    vprint = VerboseClass.vprint
    vvprint = VerboseClass.vvprint
    vvvprint = VerboseClass.vvvprint
    vvvvprint = VerboseClass.vvvvprint

    print_only = VerboseClass.print_only
    vprint_only = VerboseClass.vprint_only
    vvprint_only = VerboseClass.vvprint_only
    vvvprint_only = VerboseClass.vvvprint_only
    vvvvprint_only = VerboseClass.vvvvprint_only

    print_only_summary = VerboseClass.print_only_summary
    vprint_only_summary = VerboseClass.vprint_only_summary
    vvprint_only_summary = VerboseClass.vvprint_only_summary
    vvvprint_only_summary = VerboseClass.vvvprint_only_summary
    vvvvprint_only_summary = VerboseClass.vvvvprint_only_summary

if 0:
    ## longwinded version of the above
    def print(*p,**kw): VerboseClass.print(*p,**kw)
    def vprint(*p,**kw): VerboseClass.vprint(*p,**kw)
    def vvprint(*p,**kw): VerboseClass.vvprint(*p,**kw)
    def vvvprint(*p,**kw): VerboseClass.vvvprint(*p,**kw)
    def vvvvprint(*p,**kw): VerboseClass.vvvvprint(*p,**kw)

    def print_only(maxcount,msg,*p,**kw):
        VerboseClass.print_only(maxcount,msg,*p,**kw)
    def vprint_only(maxcount,msg,*p,**kw):
        VerboseClass.vprint_only(maxcount,msg,*p,**kw)
    def vvprint_only(maxcount,msg,*p,**kw):
        VerboseClass.vvprint_only(maxcount,msg,*p,**kw)
    def vvvprint_only(maxcount,msg,*p,**kw):
        VerboseClass.vvvprint_only(maxcount,msg,*p,**kw)
    def vvvvprint_only(maxcount,msg,*p,**kw):
        VerboseClass.vvvvprint_only(maxcount,msg,*p,**kw)

    def print_only_summary(msg,*p,**kw):
        VerboseClass.print_only_summary(msg,*p,**kw)
    def vprint_only_summary(msg,*p,**kw):
        VerboseClass.vprint_only_summary(msg,*p,**kw)
    def vvprint_only_summary(msg,*p,**kw):
        VerboseClass.vvprint_only_summary(msg,*p,**kw)
    def vvvprint_only_summary(msg,*p,**kw):
        VerboseClass.vvvprint_only_summary(msg,*p,**kw)
    def vvvvprint_only_summary(msg,*p,**kw):
        VerboseClass.vvvvprint_only_summary(msg,*p,**kw)


## BUGS (or, rather, infelicities)
##
## o There is a verbose.py that is built into anaconda
##   (albeit, not much danger of conflict)
## o pylint doesn't see vvprint() as a member of verbose class
##   because it's generated by mk_classfcn; this warning occurs
##   in every routine that imports verbose (not just pylint'ing
##   this file)
## o V_MAX_LEVEL=5 is kind of ridiculously large
##   (and arbitrary, in any case)
## o class name does not conform to PascalCase
## o class only has class methods, there are not instances
## o invocation 'from verbose import verbose as v' is confusing
##   also invoked as: 'from util.verbose import verbose as v'
##   note that: 'import util.verbose.verbose as v' fails
##   (might prefer 'import verbose as v' -- ie, as a module)
##
