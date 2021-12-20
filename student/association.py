# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        # the following only works for at most one track and one measurement
        #self.association_matrix = np.matrix([]) # reset matrix
        #self.unassigned_tracks = [] # reset lists
        #self.unassigned_meas = []
        
        #if len(meas_list) > 0:
        #    self.unassigned_meas = [0]
        #if len(track_list) > 0:
        #    self.unassigned_tracks = [0]
        #if len(meas_list) > 0 and len(track_list) > 0: 
        #    self.association_matrix = np.matrix([[0]])
        
        # unassigned tracks and measurements
        self.unassigned_tracks = []
        self.unassigned_meas = []
        N = len(track_list) # N tracks
        M = len(meas_list) # M measurements
        print('number of tracks' + str(N))
        print('number of measurements' + str(N))
        self.unassigned_tracks = np.arange(N).tolist() # unassigned tracks that will be assigned within get_closest_track_and_meas
        self.unassigned_meas = np.arange(M).tolist() # unassigned measurements that will be assigned within get_closest_track_and_meas
        
        # association matrix
        association_matrix = []
        
        # loop over all tracks and all measurements to set up association matrix
        for track in track_list:
            res = []
            for meas in meas_list:
                dist = self.MHD(track, meas, KF)
                sensor = meas.sensor
                print('track ID' + str(track.id))
                print('dist' + str(dist))
                print('meas' + str(meas))
                if self.gating(dist, sensor): # measurement lies inside the track's gate
                    res.append(dist)
                    
#                     if sensor.name == "camera":
#                         print("track {}, state={}, MHD= {}".format(track.id, track.state, MHD))
                else: # measurement lies outside the track's gate
                    res.append(np.inf) # set the distance to infinity
            association_matrix.append(res)
            
        self.association_matrix = np.matrix(association_matrix)
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # the following only works for at most one track and one measurement
        #update_track = 0
        #update_meas = 0
        
        # remove from list
        #self.unassigned_tracks.remove(update_track) 
        #self.unassigned_meas.remove(update_meas)
        #self.association_matrix = np.matrix([])
        
        A = self.association_matrix
        if np.min(A) == np.inf:
            return np.nan, np.nan

        # get indices of minimum entry
        ij_min = np.unravel_index(np.argmin(A, axis=None), A.shape) 
        ind_track = ij_min[0]
        ind_meas = ij_min[1]
        
        # delete row and column for next update
        A = np.delete(A, ind_track, 0) 
        A = np.delete(A, ind_meas, 1)
        self.association_matrix = A
        
        # update this track with this measurement
        update_track = self.unassigned_tracks[ind_track] 
        update_meas = self.unassigned_meas[ind_meas]

        # remove this track and measurement from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
        
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        
        gating_threshold = params.gating_threshold
        limit = 1./chi2.pdf(gating_threshold, df=3)
        
        if MHD <  limit:
            return True
        else:
            return False   
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        
        z_meas = np.matrix(meas.z)
        z_pred = meas.sensor.get_hx(track.x)
        gamma = z_meas - z_pred 
        
        H = meas.sensor.get_H(track.x)
        S = H * track.P * H.T + meas.R
        
        #MHD_dist = math.sqrt(gamma.T * S.I * gamma)
        MHD_dist = gamma.T * S.I * gamma
        
        return MHD_dist
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)