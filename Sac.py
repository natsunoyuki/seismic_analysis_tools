import matplotlib.pyplot as plt
import numpy as np
import obspy  
import pandas as pd
from scipy.signal import spectrogram


class Sac:
    """
    Class encapsulating the contents of a single .sac file, assuming that each file has just 1 trace.
    """
    #====================constructor====================
    def __init__(self, file_name = None):
        self.data = None
        self.df = None
        self.paz = None
        self.st = None
        self.stats = None
        if file_name is not None:
            self.read_sac(file_name)
        
    #====================pole zero file functions====================
    def read_pz_file(self, pz_file_name):
        """
        Reads the contents of a pole and zero (paz) file and saves the contents as a dictionary.
        """
        f = open(pz_file_name,'r')
        X = f.readlines()
        f.close()
        n_poles = int(X[0][6])
        poles = np.zeros(n_poles,'complex')
        for i in range(n_poles):
            buf = X[i+1]
            buf = buf.strip().split()
            poles[i] = float(buf[0]) + float(buf[1])*1j
        n_zeros = int(X[n_poles+1][6])
        zeros = np.zeros(n_zeros,'complex')
        for i in range(n_zeros):
            buf = X[i+n_poles+2]
            buf = buf.strip().split()
            if buf[0] == 'CONSTANT':
                break
            else:
                zeros[i]=float(buf[0])+float(buf[1])*1j
        constant = float(X[-1].strip().split()[1])
        
        self.paz = {"poles": poles, "zeros": zeros, "gain": constant, "sensitivity": 1}
    
    #====================.sac data functions====================
    def remove_resp(self):
        """
        Removes seismometer response from the data stream.
        """
        assert self.paz is not None
        assert self.st is not None
        self.st.simulate(paz_remove = self.paz)
        
    def read_sac(self, file_name):
        """
        Reads the contents of a .sac file containing a single trace.
        """
        self.file_name = file_name
        try:
            self.st = obspy.read(file_name, debug_headers=True)
            self.data = self.st[0].data.copy()
            self.stats = self.st[0].stats.copy()
        except Exception as e:
            print(e)
            self.st = None
            self.data = None
            self.stats = None
            
    def trim(self, start_time, end_time):
        assert self.st is not None
        if type(start_time) != obspy.UTCDateTime:
            starttime = obspy.UTCDateTime(start_time)
        if type(end_time) != obspy.UTCDateTime:
            endtime = obspy.UTCDateTime(end_time)
            
        self.st.trim(starttime, endtime) 
        self.data = self.st[0].data.copy()
        self.stats = self.st[0].stats.copy()

    def plot(self, start_time = None, end_time = None):
        assert self.st is not None
        
        if start_time is not None:
            if type(start_time) != obspy.UTCDateTime:
                start_time = obspy.UTCDateTime(start_time)
        if end_time is not None:
            if type(end_time) != obspy.UTCDateTime:
                end_time = obspy.UTCDateTime(end_time) 
        
        self.st.plot(starttime = start_time, endtime = end_time)
        plt.show()
    #====================pd.DataFrame functions====================
    def make_df(self, start_time=None, end_time=None, column_name = None):
        """
        Returns loaded data as pd.DataFrame.
        """
        if column_name is None:
            column_name = "data"
            
        assert self.data is not None
        assert self.stats is not None
  
        starttime = pd.to_datetime(str(self.stats["starttime"]))
        endtime = pd.to_datetime(str(self.stats["endtime"]))

        delta = str(self.stats["delta"]) + "S"
        datetime = pd.date_range(starttime, endtime, freq = delta)
    
        df = pd.DataFrame({"datetime": datetime, column_name : self.data})    
        df = df.set_index("datetime") # set datetime as index
       
        if start_time is None:
            start_time = starttime
        elif type(start_time) == str:
            start_time = pd.to_datetime(start_time)
        if end_time is None:
            end_time = endtime
        elif type(end_time) == str:
            end_time = pd.to_datetime(end_time)
        
        self.df = df.loc[start_time:end_time]
        return self.df

    def spectrogram(self, nperseg = None, nfft = None, detrend = False, ylim = [0, 5], cmap = "binary", plot = True):
        assert self.df is not None
        assert self.data is not None
        x = self.df.values.reshape(-1)
        f, t, Sxx = spectrogram(x = x, fs = self.stats.sampling_rate, nperseg = nperseg, nfft = nfft, detrend = detrend)
        if plot == True:
            plt.figure(figsize = (15, 5))
            plt.pcolormesh(t, f, Sxx, cmap = cmap)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.ylim(ylim)
            plt.show()
        else:
            return f, t, Sxx
    
    #====================getters and setters====================
    def get_data(self):
        """
        Returns loaded data as np.array.
        """
        return self.data    
    
    def get_df(self):
        """
        Returns loaded data as pd.DataFrame.
        """
        return self.df
    
    def get_st(self):
        """
        Returns loaded data stream as obspy.core.stream.Stream
        """
        return self.st
    
    def get_stats(self):
        """
        Returns loaded stats as obspy.core.trace.Stats.
        """
        return self.stats