classdef AmbiguityFunction < handle
    %class for the shifted ambiguity function for 2D circular pupils
    
    properties (Access = private)
        value %double[][] 2D array values
        tau %double[] scale of AF (x-direction, see T.Watson PhD Thesis) in mm
        mu %double[] scale of AF (y-direction) in mm
        OTF %double[] double array for in-focus transfer function
    end
    
    methods %constructor and get\set
        function obj = AmbiguityFunction()
        end
        function out = getValue(obj)
            out = obj.value;
        end
        function out = getOTF(obj)
            out = obj.OTF;
        end
        function out = getTau(obj)
            out = obj.tau;
        end
        function show(obj)
            figure; imagesc(obj.tau,obj.mu,real(obj.value)); xlabel('\tau = \lambda mfw (mm)');
                ylabel('mw/f(z2cos\theta-x2sin\theta) (mm)');
            axis square; title('Real-part Ambiguity Function of Circular Pupil'); drawnow;
        end
    end
    
    %% generation and access DTF's
    methods
        % generates ambiguity function based on OPTsystem and objective
        % used, fills obj.value, obj.tau and obj.mu properties
        % @param OPTSystem optSys
        % @param Objective, objective
        % @param Boolean varargin, display boolean, true==draw new figure
        function generate(obj,optSys,objective,varargin)
            %calculates shifted ambiguity function (AF_shift) for circular pupils
            nPx = 256;
            
            s_scale = 2*objective.getRadiusPP.*(-nPx/2:nPx/2-1)/(0.5*nPx);
            t_scale = objective.getRadiusPP.*(-nPx/2:nPx/2-1)/(0.5*nPx);

            %fourier aperture limits + options for focal-scannned
            if optSys.getScanBoolean==1
                focalRange = optSys.getFocalScanRange;
                rho = focalRange/objective.getF^2;
                wMax = objective.getNA*objective.getF;
                %t_scale = t_scale./focalRange*optSys.DoFinMM;
                
            else
                wMax = objective.getEffNA(optSys.getApertureRadius)*objective.getF;
            end

            
            [s,t] = meshgrid(s_scale,t_scale);
            
            obj.tau = 2*wMax.*(-nPx/2:nPx/2-1)/(0.5*nPx);
            dt = (max(t_scale)-min(t_scale))/nPx;
            mu_max = 1/(2*dt);
            obj.mu = mu_max.*(-nPx/2:nPx/2-1)/(0.5*nPx);
            
            r1 = sqrt(t.^2+s.^2);
            normFactor = zeros(size(s));
            normFactor(r1<=wMax) = 1;
            normFactor = sum(sum(normFactor.^2));
            count = 1;
            AF = zeros(length(t),length(obj.tau));

            for dt = obj.tau
                P1 = zeros(size(s)); P2 = zeros(size(t));
                r1 = sqrt((t+dt/2).^2+s.^2);
                r2 = sqrt((t-dt/2).^2+s.^2);
                P1(r1<=wMax) = 1;
                P2(r2<=wMax) = 1;
                
                if optSys.getScanBoolean==1 && rho~=0
                    gamma = dt/optSys.getLambda.*t/optSys.getRefractiveIndex;
                    switch optSys.getScanType
                        case 'linear'
                            weight = sinc(gamma*rho);
                        case 'sinusoid'
                            weight = besselj(0,pi*gamma*rho);
                        otherwise
                            error('Scan type must be linear or sinusoidal');
                    end
                    product = P1.*P2.*weight;
                else
                    product = P1.*P2;
                end
                
                A = fftshift(fft(ifftshift(product)))./normFactor;
                AF(:,count) = sum(A,2);

                count = count+1;
                if rem(count,round(length(obj.tau)/5))==0
                    disp(sprintf('AF Generation Percentage Complete: %.0f%%',(count-1)/length(obj.tau)*100));
                end
            end
            
            AF=AF./max(AF(:));
            
            obj.value = AF;
            
            obj.OTF = AF(nPx/2+1,:);
            
            if nargin==4 && varargin{1}
                figure; imagesc(obj.tau,obj.mu,real(obj.value)); 
                %xlabel('\tau = \lambda mfw (mm)'); %alternative labels
                %ylabel('mw/f(z2cos\theta-x2sin\theta) (mm)');
                xlabel('tau (mm)');
                ylabel('mu (mm)');
                axis square; title('Real-part Ambiguity Function of Circular Pupil'); drawnow;
             
                count = 1;
                [tau2D,mu2D] = meshgrid(obj.tau,obj.mu);
                zQueryRange = linspace(-optSys.traditionalDoF(objective)/2,optSys.traditionalDoF(objective)/2,nPx);
                for zPosition = zQueryRange
                    muQuery = obj.tau./(optSys.getLambda*objective.getF^2)*zPosition;
                    R(count,:) = interp2(tau2D,mu2D,obj.value,obj.tau,muQuery);
                    count = count+1;
                end
                R(isnan(R)) = 0;
                shiftedOTFz = real(R);
                freqScale = linspace(-objective.getAbbe(optSys),objective.getAbbe(optSys),nPx); 
                figure; subplot(1,3,1); plot(freqScale,real(obj.OTF)); xlabel('Spatial Frequency (mm^{-1})'); axis square; ylabel('Normalised Magnitude');
                subplot(1,3,2); imagesc(freqScale,zQueryRange,shiftedOTFz./max(shiftedOTFz(:))); axis square; xlabel('Spatial Frequency (mm^{-1})');
                ylabel('Depth away from focal plane (mm)');
                subplot(1,3,3); imagesc(freqScale,zQueryRange,repmat(obj.OTF,nPx,1)./max(obj.OTF)-shiftedOTFz./max(shiftedOTFz(:))); axis square;
                xlabel('Spatial Frequency (mm^{-1})'); ylabel('Depth away from focal plane (mm)');
            end
        end
        
        % get defocussed transfer function for OPT from AF_shift for circular pupils.
        % @param PointObject point
        % @param double theta, projection angle in radians
        % @param double zOffset, offset of focal plane from (x,z)
        % coordinate origin in mm
        % @param boolean varargin{1}, display boolean
        % @return double[] rDTF, radial defocussed transfer function
        % @return double[][] DTF, 2D defocussed transfer function
        function [DTF,rDTF] = getDTF(obj,optSys,objective,point,theta,zOffset,varargin)
            x = point.getX; z = point.getZ; 
            
            [tau2D,mu2D] = meshgrid(obj.tau,obj.mu);
            optSysType = class(optSys);
            zPosition = z*cos(theta)-x*sin(theta)-zOffset;
            switch optSysType
                case 'Standard4fSystem'
                    muQuery = obj.tau./(optSys.getLambda*optSys.getRefractiveIndex*objective.getF^2)*zPosition;
                case {'ConeBeamSystem','SubVolumeSystem'}
                    muQuery = obj.tau./(optSys.getLambda*optSys.getRefractiveIndex*objective.getF^2)*magAxial/objective.getMagnification*zPosition;
                otherwise
                    error('OPT system must be either: Standard4fSystem of ConeBeamSystem class');
            end

            rDTF = interp2(tau2D,mu2D,obj.value,obj.tau,muQuery);

            [x1,z1] = meshgrid(-size(obj.value,2)/2:size(obj.value,2)/2-1);
            DTF = interp2(x1,z1,repmat(rDTF,size(obj.value,2),1),sqrt((x1+0.5).^2+(z1+0.5).^2),(z1+0.5));
            DTF(isnan(DTF))=0;
            
            if nargin==7 && varargin{1}
                obj.show; hold on; plot(obj.tau,muQuery,'r'); hold off; drawnow;
                figure;
                imagesc(obj.tau./(optSys.getLambda*objective.getF),obj.tau./(optSys.getLambda*objective.getF),real(DTF)); 
                title('Defocussed Transfer Function'); xlabel('w (mm^{-1})'); ylabel('w (mm^{-1})'); axis square; drawnow;
            end

        end
        
        % get's image-space defocussed point spread function function for OPT from AF_shift for circular pupils.
        % @param PointObject point
        % @param double numberBesselZeros, limit of PSF scale at this
        % number of bessel zeros
        % @param double zOffset, offset of focal plane from (x,z)
        % coordinate origin in mm
        % @param boolean varargin{1}, display boolean
        % @return double[][] PSF, 2D point spread function
        % @return double[] xPsfScale, x-scale of PSF in mm
        % @return double[] yPsfScale, y-scale of PSF in mm
        % @return PointSpreadFunction psfObject, point spread function
        % object
        function [psfObject,PSF,xPsfScale,yPsfScale] = getPSF(obj,optSys,objective,point,numberBesselZeros,zOffset,method,varargin)  
            if isempty(method)
                method = 'fft';
            end

            DTF = obj.getDTF(optSys,objective,point,0,zOffset,false);
            
            %OTF = obj.getDTF(optSys,objective,PointObject(0,0,0),0,zOffset,false);
            
            if optSys.getScanBoolean==0
                NA = objective.getEffNA(optSys.getApertureRadius);
            else
                NA = objective.getNA;
            end
            
            du = 4*NA/optSys.getLambda/(size(obj.value,2)-1);
            
            switch method
                case 'fft'
                    sz = 4096;
                    pdsz = (sz-size(DTF,1))/2;
                    PSFlarge = ifftshift(ifft2(ifftshift(padarray(DTF,[pdsz,pdsz]))));
                    psf = PSFlarge(pdsz+1:pdsz+size(DTF,1),pdsz+1:pdsz+size(DTF,1));
                    uMax = du*(sz-1)/2;
                    scalePSF = (-size(DTF,1)/2:size(DTF,1)/2-1)./(2*uMax);
                    PSF = real(psf);
                case 'czt'
                    bessel_zeros = AmbiguityFunction.besselzero(1,numberBesselZeros,1);
                    x_ft_max = bessel_zeros(numberBesselZeros)/pi*optSys.getLambda/NA/2; %chirp z max frequency
                    m = 256;
                    psf = AmbiguityFunction.iczt_TW(DTF,du,x_ft_max,m);
                    %inFocusPSF(inFocusPSF<0)=inFocusPSF(inFocusPSF<0).*-1;
                    psf(psf<0)=psf(psf<0).*-1;
                    %PSF = real(psf)./sum(real(inFocusPSF(:)));
                    PSF = real(psf);
                    scalePSF = (-1:2/m:1-2/m).*x_ft_max;
                otherwise
                    error('Method must be either fft or czt');
            end
            
            optSysType = class(optSys);
            switch optSysType
                case 'Standard4fSystem'
                    xPsfScale = (point.getX+scalePSF).*objective.getMagnification; yPsfScale = (point.getY+scalePSF).*objective.getMagnification;
                case {'ConeBeamSystem','SubVolumeSystem'}
                    opCentre = optSys.getOpticCentre;
                    magRatio = optSys.magnifcationAtDepth(objective,point.getZ-zOffset)/objective.getMagnification;
                    PSF = PSF.*magRatio^2;
                    xPsfScale = ((point.getX-opCentre(1))*magRatio+opCentre(1)+scalePSF*magRatio).*objective.getMagnification;
                    yPsfScale = ((point.getY-opCentre(2))*magRatio+opCentre(2)+scalePSF*magRatio).*objective.getMagnification;
                otherwise
                    error('Incorrect system type');
            end
            
            psfObject = PointSpreadFunction(PSF,xPsfScale,yPsfScale);

            if nargin==8 && varargin{1}
                psfObject.show; title('Image-Space PSF');
            end

        end
            
    end
    
    %% static methods
    methods (Static)
        function g = iczt_TW(x,dx,fMax,m,varargin)
        %Inverse 2D inverse Chirp z-transform, given input field x with associated scale
        %increment dx, f, output size of m.
        % T.Watson edit of JC edit

        uMax = 1/(2*dx);
        fs=1;

        if fMax < uMax
                sf = fMax/uMax;
        else
                sf =1;
                fMax = uMax;
        end

        if rem(m,2)~=0
                f1 = -0.5*sf;
                f2 = 0.5*sf+sf/m;
            else
                f1 = -0.5*sf;
                f2 = 0.5*sf;
        end

        w = exp(-1i*2*pi*(f2-f1)/(m*fs));
        a = exp(1i*2*pi*f1/fs);       

        if nargin>4
            fMax2 = varargin{1};

            if fMax2 < uMax
                sf2 = fMax2/uMax;
            else
                    sf2 =1;
                    fMax2 = uMax;
            end

            if rem(m,2)~=0
                    f1b = -0.5*sf2;
                    f2b = 0.5*sf2+sf2/m;
                else
                    f1b = -0.5*sf2+sf2/m;
                    f2b = 0.5*sf2+sf2/m;
            end

            w2 = exp(1i*2*pi*(f2b-f1b)/(m*fs));
            a2 = exp(-1i*2*pi*f1b/fs);

        else
            w2 = exp(1i*2*pi*(f2-f1)/(m*fs));
            a2 = exp(-1i*2*pi*f1/fs);   
        end



        if or(iscolumn(x),iscolumn(x'))==1
                g = AmbiguityFunction.czt_JC(x,m,w,a);  
        else
                g = AmbiguityFunction.czt_JC(AmbiguityFunction.czt_JC(x,m,w2,a2).',m,w,a).';  
        end
%         
%         if or(iscolumn(x),iscolumn(x'))==1
%                 g = czt(x,m,w,a);  
%         else
%                 g = AmbiguityFunction.czt_JC(AmbiguityFunction.czt_JC(x,m,w2,a2).',m,w,a).';  
%         end

        
        end
        function g = czt_JC(x, k, w, a)
            %   James Clegg Edit for phase tilt correction. (Accomplishes same effect
            %   as fftshift in ordinary MATLAB fourier transforms). This addtional
            %   phase shift ensures that the real and imaginary components of the field
            %   overlap correctly.

            g = czt(x, k, w, a);
            %% From original CZT.m file
            [m, n] = size(x); oldm = m;
            if m == 1, x = x(:); [m, n] = size(x); end 

            %% MN and JHC correction
            pcorr = w^((-m+1)/2).^((1:k)')*(a^(k/4));
            % disp(size(pcorr))
            % disp(size(g))
            % disp(size(repmat(pcorr,1,n)))
            g = g.*repmat(pcorr,1,n);

            %% From original CZT.m file 
            if oldm == 1, g = g.'; end
        end
        function x=besselzero(n,k,kind)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            % besselzero.m
            %
            % Find first k positive zeros of the Bessel function J(n,x) or Y(n,x) 
            % using Halley's method.
            %
            % Written by: Greg von Winckel - 01/25/05
            % Contact: gregvw(at)chtm(dot)unm(dot)edu
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            k3=3*k;

            x=zeros(k3,1);

            for j=1:k3

                % Initial guess of zeros 
                x0=1+sqrt(2)+(j-1)*pi+n+n^0.4;

                % Do Halley's method
                x(j)=findzero(n,x0,kind);

                if x(j)==inf
                    error('Bad guess.');
                end

            end

            x=sort(x);
            dx=[1;abs(diff(x))];
            x=x(dx>1e-8);

            x=x(1:k);

            function x=findzero(n,x0,kind)

            n1=n+1;     n2=n*n;

            % Tolerance
            tol=1e-12;

            % Maximum number of times to iterate
            MAXIT=100;

            % Initial error
            err=1;

            iter=0;

            while abs(err)>tol && iter<MAXIT

                switch kind
                    case 1
                        a=besselj(n,x0);    
                        b=besselj(n1,x0);   
                    case 2
                        a=bessely(n,x0);
                        b=bessely(n1,x0);
                end

                x02=x0*x0;

                err=2*a*x0*(n*a-b*x0)/(2*b*b*x02-a*b*x0*(4*n+1)+(n*n1+x02)*a*a);

                x=x0-err;
                x0=x;
                iter=iter+1;

            end

            if iter>MAXIT-1
                warning('Failed to converge to within tolerance. ',...
                        'Try a different initial guess');
                x=inf;    
            end
            end
        end
    end
    
end

