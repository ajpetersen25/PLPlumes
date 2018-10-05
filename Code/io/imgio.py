from __future__ import division
import numpy as np

class imgio: 
    '''
    Interface class for the img files 
    
    Requires 1 argument, filename. If filename exists,
    it open the file, reads in header information and populates
    the class instance variables.
    If file does not exist, it will write to a new file.
    Typical usage:
    a = imgio.imgio('existing.img')
    print a.it # number of frames
    
    # plot the last column of the last frame using matplotlib.pyplot (plt) and numpy (np)
    plt.imshow(np.reshape(a.read_frame(a.it),(a.iy,a.ix)),cmap=plt.cm.Greys_r)
    See pivio repo documentation for examples
    '''
    def __init__(self, file_name=None):
        if file_name != None:
            self.file_name = file_name
            try:
                with open(file_name): 
                    self.header = self._read_header()
                    self.frame_length = self.ix*self.iy*self.bytes
                    self.comment = self._read_comment()
                    if self.bytes == 1:
                        self.type = 'uint8'
                    elif self.bytes == 2:
                        self.type = 'uint16'
            except IOError:
                print 'No file found: %s' % self.file_name
                print 'This instance will create new file'

    def __str__(self):
        ''' This is returned when you write >> print m # where m is an instance '''
        
        s = 'imgio instance of file %s stored at %s\n\n' \
            'IMG data:\n\n' \
            'ix:       %d\n'  \
            'iy:       %d\n'  \
            'it:       %d\n'  \
            'min:      %.0f\n'\
            'max:      %.0f\n'\
            'unsigned: %d\n'  \
            'bytes:    %d\n'  \
            'channels: %d\n' % (self.file_name,hex(id(self)),self.ix,self.iy,
                                self.it,self.min,self.max,self.unsigned,self.bytes,self.channels)

	return s

    def __repr__(self):
        ''' This is returned when you write >> m # where m is an instance '''
         
	return 'imgio("%s",ix=%d,iy=%d,it=%d,min=%.0f,' \
            'max=%.0f,unsigned=%d,bytes=%d,channels=%d)' % (self.file_name,self.ix,self.iy,
                                                            self.it,self.min,self.max,self.unsigned,
                                                            self.bytes,self.channels)

    def _read_header(self):
        """ Read the header information """
        with open("%s" % self.file_name, "rU") as f:
            byte = f.read(1)
            while byte != '#':
                byte = f.read(1)
                self.comment_length = f.tell()
            f.readline()
            self.ix, self.iy, self.it = map(int, f.readline().split())
            self.min, self.max = map(float, f.readline().split())
            self.unsigned, self.bytes, self.channels = map(int, f.readline().split())
            self.header_length = f.tell()
            self.file_pos = self.header_length
            self.comment = ''

    def write_header(self):
        """ Write the header information """
        with open("%s" % self.file_name, "w") as f:
            f.write('%s#\n' % self.comment)
            self.comment_length = f.tell()
            f.write('%d %d %d\n' % (self.ix, self.iy, self.it))
            f.write('%f %f\n' % (self.min, self.max))
            f.write('%d %d %d\n' % (self.unsigned, self.bytes, self.channels))
            self.header_length = f.tell()
            self.file_pos = self.header_length
            self.frame_length = self.ix*self.iy*self.bytes
         
    def read_frame(self,frame_number):
        """ Read a frame number from the file """
        with open("%s" % self.file_name, "rb") as f:
            f.seek(self.header_length+self.frame_length*frame_number)
            byte = np.fromstring(f.read(self.ix*self.iy*self.bytes),dtype=self.type) 
            self.file_pos = f.tell()
        return byte
        
    def read_frame2d(self,frame_number,noflip=False):
        '''
        Return a 2D img frame flipped for plotting
        
        Usage: a = img.read_frame2d(0)
        
        Arguments
           frame_number  - int   The frame number to retrieve
           noflip            - bool - set to True not to do a vertical flip of array (for plotting)
        Returns
           frame         - 2D np.array - A two dimensional numpy array
        '''
        if noflip:
            return self.read_frame(frame_number).reshape(self.iy,self.ix)
        else:
            return np.flipud(self.read_frame(frame_number).reshape(self.iy,self.ix))
        

    def read_subframe(self,i,x_start,y_start,width,height):
        """
        Read a subimage from a frame based on a start location and width/height
        xy_start is a tuple specifying the start of the rectangle (top left)
        width and height are pixels dimensions of rectangle to return
        """
        print 'WARNING: This will be removed'
        with open("%s" % self.file_name, "rb") as f:
            # find xstart,ystart
            try:
                byte = np.zeros(width*height)
            except:
                print 'cannot allocate array size of %dx%d' % (width,height)
            for j in np.arange(height):
                # do absolute position (relateive takes '1' as a second argument)
                # speeds should be the same?
                f.seek(self.header_length+self.frame_length*i+self.ix*(y_start+j)+x_start)
                byte[j*width:(j+1)*width] = np.fromstring(f.read(width),dtype=self.type)
            self.file_pos = f.tell()
        return np.reshape(byte,(height,width))

    def write_frame(self,data):
        """ 
        Append a frame to the file 
        Usage: img.write_frame(data)        
        
        Arguments
            data      - 2D int array - An integer array of size img.iy,img.ix
            
        Returns
            NA
            
        Notes: Does not check for a valid IMG file
        """
        # TODO - ensure a header exists
        with open("%s" % self.file_name, "ab") as f:
            f.write(data.astype(self.type))

    def _read_comment(self):
        """ Read the comment from the pivio file """
        with open("%s" % self.file_name, "rU") as f:
            byte =  f.read(self.comment_length-1)
        return byte

    def sub2ind(self,x,y):
        """ Return the index of a frame based on x,y coordinates
        i = sub2ind(x,y)
        """
	return y*self.ix+x

    def ind2sub(self,i):
        """ Return the (x,y) coordinates of a frame based on the
        frame index
        x,y = ind2sub(i)
        """
        x = i%self.ix
        y = np.floor(i/self.ix)
return x,y
