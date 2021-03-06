from __future__ import division
import numpy as np

class pivio: 
    '''
    Interface class for the piv files 
    
    Requires 1 argument, filename. If filename exists,
    it open the file, reads in the comments and populates
    the class instance variables.
    If file does not exist, it will write to a new file.
    Typical usage:
    a = pivio.pivio('existing.piv')
    print a.nt # number of frames
    
    # plot the last column of the last frame
    pcolor(numpy.reshape(a.read_frame(a.nt)[a.cols],(a.ny,a.nx)))
    See pivio documentation for examples
    '''
    def __init__(self, file_name=None, **kwargs):

        if file_name != None:
            self.file_name = file_name
            try:
                with open(file_name): 
                    self.header = self._read_header()
                    self.frame_length = self.nx*self.ny*self.cols
                    self.vecs = self.nx*self.ny
                    self.comment = self._read_comment()
            except IOError:
                print 'Creating file: %s' % self.file_name
                self.init(**kwargs)

    def init(self, **kwargs):
        ''' 
        Initialise a new piv file 
        Helper function to allow the initialisation of the header for a new piv file.
        '''
        # initialise variables to defaults
        arg_vals = { 'cols':3, 'ix':0, 'iy':0, 'nx':0, 'ny':0, 'nt':0, 'x0':0, 'y0':0, 't0':0,
                     'dx':0, 'dy':0, 'dt':1, 'sx':1.0, 'sy':1.0, 'st':1.0, 'rx':0.0, 'ry':0.0, 'rt':0.0,
                     'comment':''}

        # update default dict with keyword arguments
        arg_vals.update(kwargs)
        # assign updated dict of keywords to local instance as variables
        self.__dict__.update(arg_vals)
        #same as above line
        #for keyword,arg in arg_vals.iteritems():
        #    setattr(self,keyword,arg)

    def __str__(self):
        ''' This is returned when you write >> print m # where m is an instance '''
        s = 'pivio instance of file %s stored at %s\n\n' \
            'PIV data:\n\n' \
            'Image information\n'\
            'ix:       %d\n'\
            'iy:       %d\n\n'\
            'Vector information\n'\
            'nx:       %d\n'\
            'ny:       %d\n'\
            'nt:       %d\n'\
            'x0:       %d\n'\
            'y0:       %d\n'\
            't0:       %d\n'\
            'dx:       %d\n'\
            'dy:       %d\n'\
            'dt:       %d\n\n'\
            'Column information\n'\
            'cols:     %d\n\n'\
            'Scale factors\n'\
            'sx:       %.1f\n'\
            'sy:       %.1f\n'\
            'st:       %.1f\n\n'\
            'Real space origin\n'\
            'rx:       %.1f\n'\
            'ry:       %.1f\n'\
            'rt:       %.1f\n\n'\
            'Read comment using <pivio instance>.comment \n'  % (self.file_name,hex(id(self)),self.ix,self.iy,
                                                                 self.nx,self.ny,self.nt,self.x0,self.y0,self.t0,self.dx,self.dy,self.dt,
                                                                 self.cols,self.sx,self.sy,self.st,self.rx,self.ry,self.rt)
	return s

    def __repr__(self):
        ''' This is returned when you write >> m # where m is an instance '''
        
        return 'pivio("%s",ix=%d,iy=%d,nx=%d,ny=%d,' \
            'nt=%d,cols=%d)' % (self.file_name,self.ix,self.iy,
                                self.nx,self.ny,self.nt,
                                self.cols)

    def _read_header(self):
        """ Read the header information """
        with open("%s" % self.file_name, "rU") as f:
            byte = f.read(1)
            self.comment_length = 0
            while byte != '#':
                byte = f.read(1)
                self.comment_length = f.tell()
            f.readline()
            self.ix, self.iy = map(int, f.readline().split())
            self.nx, self.ny, self.nt = map(int, f.readline().split())
            self.x0, self.y0, self.t0 = map(int, f.readline().split())
            self.dx, self.dy, self.dt = map(int, f.readline().split())
            self.sx, self.sy, self.st = map(float, f.readline().split())
            self.rx, self.ry, self.rt = map(float, f.readline().split())
            self.cols = int(f.readline())
            self.header_length = f.tell()
            self.file_pos = self.header_length
            self.comment = ''

    def write_header(self):
        """ Write the header information """
        with open("%s" % self.file_name, "w") as f:
            f.write('%s#\n' % self.comment)
            self.comment_length = f.tell()
            f.write('%d %d\n' % (self.ix, self.iy))
            f.write('%d %d %d\n' % (self.nx, self.ny, self.nt))
            f.write('%d %d %d\n' % (self.x0, self.y0, self.t0))
            f.write('%d %d %d\n' % (self.dx, self.dy, self.dt))
            f.write('%f %f %f\n' % (self.sx, self.sy, self.st))
            f.write('%f %f %f\n' % (self.rx, self.ry, self.rt))
            f.write('%d\n' % self.cols)
            self.header_length = f.tell()
            self.file_pos = self.header_length
            self.frame_length = self.nx*self.ny*self.cols
         
    def read_frame(self,frame_number):
        """ 
        Read a frame number from the file 
        Usage: piv.read_frame(0)
        Arguments
            frame_number      - int - Frame number to retrieve
            
        Returns
            frame             - list of 1D arrays - list of len = piv.cols containing arrays of len piv.ny x piv.nx
        """
        with open("%s" % self.file_name, "rb") as f:
            f.seek(self.header_length+4*self.frame_length*frame_number)
            byte = [ np.fromstring(f.read(4*self.nx*self.ny),dtype='>f4') for j in range(self.cols)]
            self.file_pos = f.tell()
        return byte
        
    def s(self,frame_number,noflip=False):
        """ 
        Read velocity status from frame number from the file 
        Usage: piv.v(0)
        Arguments
            frame_number      - int - Frame number to retrieve
            noflip            - bool - set to True not to do a vertical flip of array (for plotting)
            
        Returns
            frame             - s velocity of shape (piv.ny,piv.nx)
        """
        
        if noflip is True:
            return self.read_frame(frame_number)[0].reshape(self.ny,self.nx)
        else:
            return np.flipud(self.read_frame(frame_number)[0].reshape(self.ny,self.nx))
            
    def u(self,frame_number,noflip=False):
        """ 
        Read velocity u from frame number from the file 
        Usage: piv.u(0)
        Arguments
            frame_number      - int - Frame number to retrieve
            noflip            - bool - set to True not to do a vertical flip of array (for plotting)
            
        Returns
            frame             - u [1] velocity of shape (piv.ny,piv.nx)
        """
        if noflip:
            return self.read_frame(frame_number)[1].reshape(self.ny,self.nx)
        else:
            return np.flipud(self.read_frame(frame_number)[1].reshape(self.ny,self.nx))

           
    def v(self,frame_number,noflip=False):
        """ 
        Read velocity v from frame number from the file 
        Usage: piv.v(0)
        Arguments
            frame_number      - int - Frame number to retrieve
            noflip            - bool - set to True not to do a vertical flip of array and negation (for plotting)
            
        Returns
            frame             - flipped and negated v velocity of shape (piv.ny,piv.nx)
        """
        
        if noflip is True:
            return self.read_frame(frame_number)[2].reshape(self.ny,self.nx)
        else:
            return -np.flipud(self.read_frame(frame_number)[2].reshape(self.ny,self.nx))
        
    def read_frame2d(self,frame_number,noflip=False):
        """ 
        Read all column from frame number from the file 
        Usage: piv.read_frame2d(0)
        Arguments
            frame_number      - int - Frame number to retrieve
            noflip            - bool - set to True not to do a vertical flip of array and negation (for plotting)
            
        Returns
            frame             - list of columns each of shape (piv.ny,piv.nx)
        """
        frame = self.read_frame(frame_number)
        if noflip is True:
            frame2d = [frame[i].reshape(self.ny,self.nx) for i in xrange(self.cols)]
        else:
            frame2d = [np.flipud(frame[i].reshape(self.ny,self.nx)) for i in xrange(self.cols)]
            frame2d[2] = -frame2d[2]

        return frame2d

    @property
    def x(self):
        '''
        Return the x grid locations
        '''
        return np.arange(self.x0,self.nx*self.dx+self.x0,self.dx)

    @property
    def y(self):
        '''
        Return the x grid locations
        '''
        return np.arange(self.y0,self.ny*self.dy+self.y0,self.dy)

    def write_frame(self,data):
        """ 
        Append a frame to a piv file
        Usage: piv.write_frame(data)
        Arguments
            data             - list of np.array - List of len piv.cols of arrays of piv.ny x piv.nx entries
            
        Returns
            NA
        """
        # TODO - ensure a header exists
        with open("%s" % self.file_name, "ab") as f:
            if len(data) == self.cols:
                [ f.write(data[i].astype('>f4')) for i in range(self.cols)]
            else:
                f.write(data.astype('>f4'))

    def _read_comment(self):
        """ Read the comment from the pivio file """
        with open("%s" % self.file_name, "rU") as f:
            byte =  f.read(self.comment_length-1)
        return byte

    def sub2ind(self,x,y):
        """ Return the index of a frame based on x,y coordinates
        i = sub2ind(x,y)
        """
	return y*self.nx+x

    def ind2sub(self,i):
        """ Return the (x,y) coordinates of a frame based on the
        frame index
        x,y = ind2sub(i)
        """
        x = i%self.nx
        y = np.floor(i/self.nx)
        return x,y
