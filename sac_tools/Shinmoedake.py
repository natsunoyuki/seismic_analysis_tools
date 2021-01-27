import numpy as np
import pandas as pd
import obspy 
import os

class Shinmoedake:
    """
    Class encapsulating Shinmoedake .sac file data, and for converting the .sac data into pd.DataFrames
    for a more pythonic way of dealing with the data.
    
    This class is specifically tailored to how the Shinmoedake data is stored for the 2011 eruption.
    Each day's data is stored in 1 daily directory.
    Within each daily directory, each hour's data is stored in 1 hourly directory 
    (24 hourly directories per day).
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir
    
    def datetime_to_parent_dir(self, datetime = "110202"):
        """
        Converts datetime to parent directory such as 110202.

        Inputs
        ------
        datetime: datetime object or str
            Timestamp of the wanted file.

        Returns
        -------
        directory: str
            corresponding directory name
        """
        if type(datetime) == str:
            datetime = pd.to_datetime(datetime)

        parent_dir = str(datetime.year)[2:] + str(datetime.month).zfill(2) + str(datetime.day).zfill(2)
        return parent_dir    
    
    def datetime_to_directory(self, datetime = "110202 20:00:00"):
        """
        Converts datetime to directory such as 110202/11020220h.

        Inputs
        ------
        datetime: datetime object or str
            Timestamp of the wanted file.

        Returns
        -------
        directory: str
            corresponding directory name
        """
        if type(datetime) == str:
            datetime = pd.to_datetime(datetime)

        directory = self.datetime_to_parent_dir(datetime)
        directory = os.path.join(directory, directory + str(datetime.hour).zfill(2) + "h")
        return directory
    
    def read_sac(self, station = "EV.SMN", channel = "wU", start_time = "20110202 20:43:00", end_time = "20110202 20:55:00"):
        """
        Reads data from .sac files, starting from start_time to end_time.
        This function is tailored specifically for the format which Shinmoedake
        seismograms are stored.
        Each day's data is stored in 1 daily directory.
        Within each daily directory, each hour's data is stored in 1 hourly directory 
        (24 hourly directories per day).
        
        Inputs
        ------
        station: str
            station name e.g. "EV.SMN"
        channel: str
            channel name e.g. "wU"
        start_time: datetime object or str
            start_time to start loading
        end_time: "20110202 20:43:00"
            end_time to finish loading
        
        Returns
        -------
        st: obspy.Stream
            loaded .sac data
        """
        base_dir = self.base_dir
        # First, check how many daily directories (110126, 110127...) are spanned.
        # Then for each daily directory, check how many hourly sub-dirs (11012600h, 11012601h...) are spanned.
        if type(start_time) == str:
            start_time = pd.to_datetime(start_time)
        if type(end_time) == str:
            end_time = pd.to_datetime(end_time)
            
        hourly_dirs = pd.date_range(start_time.floor("H"), end_time.floor("H"), freq = "1H")
        
        if len(hourly_dirs) == 1:
            directory = self.datetime_to_directory(start_time)
            directory = os.path.join(base_dir, directory, station + "." + channel)
            st = obspy.read(directory, debug_headers = True)
            st = self.trim(st, start_time, end_time)
        else:
            for count, hour in enumerate(hourly_dirs):
                directory = self.datetime_to_directory(hour)
                directory = os.path.join(base_dir, directory, station + "." + channel)
                if count == 0:
                    st = obspy.read(directory, debug_headers = True)
                    st = self.trim(st, start_time, start_time.ceil("H"))
                elif count == len(hourly_dirs) - 1:
                    tmp_st = obspy.read(directory, debug_headers = True)
                    tmp_st = self.trim(tmp_st, end_time.floor("H"), end_time)
                    st = st + tmp_st
                else:
                    tmp_st = obspy.read(directory, debug_headers = True)
                    st = st + tmp_st
        return st
    
    def st_to_df(self, st, column_name = "data"):
        """
        Converts obspy.Stream to pd.DataFrame.
        
        Inputs
        ------
        st: obspy.Stream
        
        Returns
        -------
        df: pd.DataFrame
        """
        df = None
        
        for i in range(len(st)):
            start_time = pd.to_datetime(str(st[i].stats["starttime"]))
            end_time = pd.to_datetime(str(st[i].stats["endtime"]))
            delta = str(st[i].stats["delta"]) + "S"
            datetime = pd.date_range(start_time, end_time, freq = delta)
            _df = pd.DataFrame({"datetime": datetime, column_name : st[i].data})    
            _df.set_index("datetime", inplace = True) # set datetime as index
            if df is None:
                df = _df.copy()
            else:
                df = pd.concat([df, _df])
        return df

    def trim(self, st, start_time, end_time):
        """
        Trims a seismograph trace from start_time to end_time.
        
        Inputs
        ------
        st: obspy.Stream
            seismograph to trim
        start_time: datetime object or string
        end_time: datetime object or string
        
        Returns
        -------
        st: obspy.Stream
            trimmed seismograph
        """
        if type(start_time) != obspy.UTCDateTime:
            starttime = obspy.UTCDateTime(start_time)
        if type(end_time) != obspy.UTCDateTime:
            endtime = obspy.UTCDateTime(end_time)
        st.trim(starttime, endtime) 
        return st