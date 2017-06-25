classdef PointSpreadFunction < handle
    %Class with basic PSF properties, value, scale and methods to calculate
    %FWHM from gaussian fitting. Method to ramp-filter point spread
    %function
    
    properties (Access=private)
        value %double[][] 2D array of PSF values
        xScale %x-scale in mm
        yScale %y-scale in mm
        FWHM %full-width half maximum spatial resolution in mm
        rampValue %double[][] 2D array of ramp filtered PSF
    end
    
    methods %constructor and get/set
        function obj = PointSpreadFunction(varargin)
            if nargin==3
                obj.value = varargin{1};
                obj.xScale = varargin{2};
                obj.yScale = varargin{3};
            elseif nargin~=0
                error('Send value,xScale and yScale or no arguments');
            end
        end
        % @return double out, fwhm resolution
        function out = getFWHM(obj)
            if isempty(obj.FWHM)
                obj.calculateFWHM;
            end
            out = obj.FWHM;
        end
        % @return double[] out, x-scale in mm
        function out = getXScale(obj)
            out = obj.xScale;
        end
        % @return double[] out, y-scale in mm
        function out = getYScale(obj)
            out = obj.yScale;
        end
        % @return double[][] out, value of psf
        function out = getValue(obj)
            out = obj.value;
        end
        % @param double[][] value, value of psf
        function setValue(obj,value)
            obj.value = value;
        end
        % @param double[] xScale,x-scale in mm
        function setXScale(obj,xScale)
            obj.xScale = xScale;
        end
        % @param double[] yScale, y-scale in mm
        function setYScale(obj,yScale)
            obj.yScale = yScale;
        end
        % @param double[][] rampValue, rampValue filtered psf
        function setRampValue(obj,rampValue)
            obj.rampValue = rampValue;
        end
        % @return double[][] varargout, optional returns value
        function varargout = show(obj)
            figure;
            imagesc(obj.xScale,obj.yScale,obj.value); title('PSF'); xlabel('(mm)'); axis square;
            varargout{1} = obj.value;
        end
        % @return complex[][] varargout, optional returns ramp value
        function varargout = showRamp(obj)
            if ~isempty(obj.rampValue)
                figure;
                imagesc(obj.xScale,obj.yScale,real(obj.rampValue)); title('PSF'); xlabel('(mm)'); axis square;
                varargout{1} = obj.rampValue;
            else
                error('Please calculate ramp value first');
            end
        end
        % @param int rowIndex, gets row profile of rowIndex
        % @return double[] out, psf row profile
        function out = rowProfile(obj,rowIndex)
            out = obj.value(rowIndex,:);
        end
         % @param int colIndex, gets column profile of rowIndex
        % @return double[] out, psf column profile
        function out = colProfile(obj,colIndex)
            out = obj.value(:,colIndex);
        end
        
    end
    
    methods
        % @param boolean varargin, display boolean, true==new figure
        function out = calculateFWHM(obj,varargin)
            if ~isempty(obj.value) && ~isempty(obj.xScale) && ~isempty(obj.yScale)
                if nargin==2 && varargin{1}
                    [obj.FWHM,gaussianFit] = PointSpreadFunction.fit2DGaussian(obj.value,obj.xScale,obj.yScale,true);
                else
                    [obj.FWHM,gaussianFit] = PointSpreadFunction.fit2DGaussian(obj.value,obj.xScale,obj.yScale);
                end
                figure; plot(obj.xScale,obj.value(size(obj.value,1)/2+1,:)); hold on; plot(obj.xScale,gaussianFit(size(obj.value,1)/2+1,:)); 
                xlabel('x (mm)'); ylabel('Intensity'); hold off;  title('1D profile through fit');drawnow;
                out = obj.getFWHM;
            else
                error('Please set PSF value and scales before calculating fwhm');
                out = [];
            end
        end
        function rampFilter(obj)
            if ~isempty(obj.value)
                nPx = size(obj.value,2);
                ramp = repmat((abs(-nPx/2:nPx/2-1)*2/nPx),size(obj.value,1),1);
                obj.rampValue = ifftshift(ifft2(ifftshift(fftshift(fft2(fftshift(obj.value))).*ramp)));
            else
                error('Please set PSF value before calculating ramp filter');
            end
        end
        % @return double[] rPSF, returns x-profile of psf
        % @return double[] scale, scale in mm
        % @return int row, row of profile
        function [xPSF,scale,row] = xProfile(obj)
            if ~isempty(obj.value)
                [row,~] = find(obj.value == max(obj.value(:)));
                xPSF = obj.value(row,:);
                scale = obj.xScale;
            else
                error('No PSF value');
            end
        end
        % @return double sum, sum of PSF
        function out = sum(obj)
            out = sum(obj.value(:));
        end
    end

    methods (Static)
        % @param double[][] xydata, 2D image data to fit single 2D gaussian
        % @param double[] xScale, x-scale of xydata in mm
        % @param double[] yScale, y-scale of xydata in mm
        % @param boolean varargin, display boolean, true==new figure
        % @return double FWHM, full-width half maximum spatial resolution in mm
        % @return double[][] gaussianFit
        function [FWHM,gaussianFit] = fit2DGaussian(xydata,xScale,yScale,varargin)
            % T.Watson based on fminsearch model
            [xdata,ydata] = meshgrid(xScale,yScale);
            [row,col] = find(xydata==max(xydata(:)));

            row = round(mean(row));
            col = round(mean(col));
            
            amplitudeGuess = xydata(row,col);
            sigmaGuess = (max(xScale)-min(xScale))/100;
            x0guess = xdata(row,col);
            y0guess = ydata(row,col);
            offsetGuess = std(xydata(:));

            start_guesses = [amplitudeGuess,0,sigmaGuess,sigmaGuess,x0guess,y0guess,offsetGuess];
            
            if nargin==4 && varargin{1}
                drawBoolean = true;
                figure;
            else
                drawBoolean = false;
            end
            
            model = @fun;
            options = optimset('MaxFunEvals',2000*length(start_guesses),'MaxIter',2000*length(start_guesses));
            estimates = fminsearch(model,start_guesses,options);
            [~,gaussianFit] = model(estimates);
            
            FWHM = (estimates(3)+estimates(4))/2*2*sqrt(2*log(2));

            function [sse,fit] = fun(params)
                A = params(1);
                theta = params(2);
                sigx = params(3);
                sigy = params(4);
                x0 = params(5);
                y0 = params(6);
                B = params(7);

                a = cos(theta)^2/2/sigx^2+sin(theta)^2/2/sigy^2;
                b = -sin(2*theta)/4/sigx^2+sin(2*theta)/4/sigy^2;
                c = sin(theta)^2/2/sigx^2+cos(theta)^2/2/sigy^2;

                fit = A.*exp(-(a.*(xdata-x0).^2+2*b.*(xdata-x0).*(ydata-y0)+c.*(ydata-y0).^2))+B;
                if drawBoolean
                    subplot(1,2,1)
                    imagesc(xScale,yScale,xydata); axis square;
                    subplot(1,2,2)
                    imagesc(xScale,yScale,fit); axis square;
                    drawnow;
                end      
                sse = sum(sum((fit-xydata).^2));
            end
        end
    end
    
end

