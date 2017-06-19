classdef ObjectiveStage < handle
    %Class representing translation of the optical system along the z-axis
    
    properties (Access = private)
        motion % double[] z-values for tracking, length = OPTSystem.getNProj
        calibrationImagesPath % string, path to folder of calibration images (square array of beads at 180 degree to each other)
        aorLocation %double, location of the axis of rotation in mm relative to the objective stage scale
    end
    
    methods 
        % CONSTRUCTOR
        function obj = ObjectiveStage()
        end
        
        % GET/SET
        function out = getMotion(obj)
            out = obj.motion;
        end
        function out = getCalibPath(obj)
            out = obj.calibrationImagesPath;
        end
        % @param string path, path to folder containing calibration images
        function setCalibPath(obj,path)
            obj.calibrationImagesPath = path;
        end
        % @param double location, location of the aor relative to the
        % objective stage in mm
        function setAoRLocation(obj,location)
            obj.aorLocation = location;
        end
        
        function out=getAoRLocation(obj)
            out = obj.aorLocation;
        end
        
        function plotMotion(obj)
            figure; plot(obj.motion);
        end
        
        % @param double[] motion, 1D array of motion
        function setMotion(obj,motion)
            obj.motion = motion;
        end
        
        % fills motion with array of z-values for sinusoidal motion around
        % centre of OPTSystem volume
        % @param OPTSystem optSys, theta values
        % @param double amplitude, amplitude of motion in mm (object space)
        % @param double phase, phase of motion in radians
        % @param double varargin, optional offset of motion in mm
        function sinusoidalMotion(obj,optSys,amplitude,phase,varargin)
            if nargin==5
                offset = varargin{1};
            else
                offset = 0;
            end
            obj.motion = amplitude.*sin(optSys.theta+phase)+offset;
        end
        
        % @param OPTSystem optSys
        % @param StepperMotor stepperMotor
        % @param PointObject point
        function trackPoint(obj,optSys,point)
            obj.motion = zeros(1,optSys.getNProj);
            for idx = 1:optSys.getNProj
                [~,~,~,z(idx)] = optSys.stepperMotor.rotate(point,idx,optSys.theta);
            end
            obj.motion = z;
            obj.plotMotion;
        end
        
        % @param int projectionNumber, projection number
        % @return double out, z-position of objective stage in mm
        function out = getMotionAtProj(obj,projectionNumber)
            if isempty(obj.motion)
                out = 0;
            else
                out = obj.motion(projectionNumber);
            end
        end
           
    end
    
end

