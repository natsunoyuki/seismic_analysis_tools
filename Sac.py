import numpy as np
import pandas as pd
import obspy  

class Sac:
    """
    Class encapsulating the contents of a single .sac file, 
    assuming that each file has just 1 trace.
    """
    def __init__(self, file_name):
        self.read_sac(file_name)
        self.df = None
            
    def read_sac(self, file_name):
        self.file_name = file_name
        try:
            self.st = obspy.read(file_name, debug_headers=True)
            self.data = self.st[0].data
            self.stats = self.st[0].stats
        except Exception as e:
            print(e)
            self.st = None
            self.data = None
            self.stats = None

    def make_df(self, start_time=None, end_time=None, column_name = None):
        """
        Returns loaded data as pd.DataFrame.
        """
        if column_name is None:
            column_name = "data"

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