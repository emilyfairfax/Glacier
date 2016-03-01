% Glacier Mass Balance and Flow Model: Oscillating ELA with Concave
% Topography, Exaggerated Erosion, and Ice Tributary

% written by Emily Fairfax
% February 20th, 2016
% with plotting guidance from Megan Brown


%% Initialize
clear all

% Constants
    % Topography Constants
    topomax = 6000; % max elevation of valley in meters
    m = 1; % slope of valley (for linear topography)
   
    % Mass Balance Constants
    gamma = 0.01; % b slope in yr-1
    ampELA=500; % half amplitude of sin function for varying climate, how much elevation does ELA change by +/-
    pELA = 250; % period of climate oscillations in years
    startELA = (5000); % equilibrium line altitude (ELA) mid in meters
    ELA = startELA; % Initial ELA

    % Main Glacier Flux Constants
    Uo=60; % glacier sliding m/year
    Hstar = 60; % critial thickness in meters
    A = 10^-16; % Flow law parameter in Pa-3 yr-1
    rhoi = 917; % density of ice in kg/m3
    g = 9.81; % gravitational acceleration
    valleywidth = 200; % width of valley holding the glacier in meters
    
    
    % Erosion Constants
    a = 5*10^-4;
    n = 2;
        
% Time and Space
    % Time Array
    dt = 0.01; % time step in years
    tmax = 2000; % tmax in years
    t = 0:dt:tmax; % time array
    imax = length(t); % loop length
    nplot = 100; % number of plots
    tplot = tmax/nplot; % times at which plots will be made

    % Space Array
    dx = 100; % horizontal step in meters
    xmin = 0; % min horizontal space in meters
    xmax = topomax/m; % max horizontal space in meters
    x = xmin:dx:xmax; % x array
    xsq = x.*x;
    xedge = xmin+(dx/2):dx:xmax-(dx/2); % xedge array

%Variable Arrays
    %Nodes
    N = ((xmax/dx))+1; % number of nodes, based on space array

    % Bedrock Topography
    %zrock = topomax - m.*x; % linear valley profile
    zrock1 = 3000+(topomax-3000)*exp(-.5*m*10^-3*x);
    tzrock1 = 3000+(topomax-3000)*exp(-.5*m*10^-3*x);
    
    % Junction Info
    xjunction = 1200; %where along the main glacier profile is the junction in physical space
    junctionxarray = xjunction/dx; %find that location in the x array
    zjunction = zrock1(junctionxarray); %find the elevation of the junction (useful info for deciding climate parameters, etc)


    % Glacier Variables and Arrays
    icethickness1 = zeros(1,N); % glacier thickness, initial condition is no ice
    iceelevation1 = zrock1+icethickness1; % ice elevation array
    iceslope = diff(iceelevation1)./dx; % array of slope based on ice elevations
    edgeicethickness1 = zeros(1,N-1); % array for ice thicknesses at the edges
    jmax = length(edgeicethickness1); % length for loop to fill ice thickness at edges
    for j=1:1:jmax
        edgeicethickness1(j) = (icethickness1(j)+icethickness1(j+1))/2; % calculate thickness of ice at edges by averaging boxes 
    end
        
    %Tributary
    ticethickness1 = zeros(1,N); % glacier thickness, initial condition is no ice
    ticeelevation1 = zrock1+ticethickness1; % ice elevation array
    ticeslope = diff(ticeelevation1)./dx; % array of slope based on ice elevations
    tedgeicethickness1 = zeros(1,N-1); % array for ice thicknesses at the edges
    jmax = length(tedgeicethickness1); % length for loop to fill ice thickness at edges
    for j=1:1:jmax
        tedgeicethickness1(j) = (ticethickness1(j)+ticethickness1(j+1))/2; % calculate thickness of ice at edges by averaging boxes 
    end
    % Mass Balance
    b = gamma*(iceelevation1-ELA); % mass balance, b, array
    
        
    % Ice Flux via Flow Law
    QflowMain = zeros(1,N+1); % inital condition for ice flux by the flow law is no flux
    QflowMainOriginal = zeros(1,N+1); %array for use in tributary flows
    QflowTrib = zeros(1,N+1); % create an array for the ice tributary flow
    QaddTrib = zeros(1,N); % create an array to house the flow addition from the tributary along the profile of the main glacier
        

% For Plotting Make Arrays of Specific Sizes

%For Plotting x Axis
xplot = zeros(1,N+1); %for use in elevation and thickness plots
xplot(1)=-1; %for use in elevation and thickness plots
xplot(2:N+1)=x; %for use in elevation and thickness plots
xplot2=xmin:dx:xmax+dx; %for use in Q plot

%For Plotting Ice Elevation
iceplot = zeros(1,N+1);
iceplot(1)=topomax;
iceplot(2:N+1)=iceelevation1;

%For Plotting Bedrock Elevation
rockplot = zeros(1,N+1);
rockplot(1)=0;
rockplot(2:N+1)=zrock1;

%For Plotting Ice Thicknesses
thicknessplot = zeros(1,N+1);
thicknessplot(1)=0;
thicknessplot(2:N+1)=icethickness1;
thickplot =0:10:500; % array of large elevation values
junctionforthick = ones(size(thickplot))*(xjunction); % make a line at x = 0 for y values in zplot

%For Plotting Mass Balance Line
zplot =0:1000:10000; % array of large elevation values
zerosforb = zeros(size(zplot)); % make a line at x = 0 for y values in zplot

%For Filling Plots
bottomline1=zeros(N+1:2); %%Create a bottom line array for use in filling plots, needs to be same length as N
bottomline1(2:N+1)=-20000000; %arbitrary very negative number
bottomline2=zeros(N:2); %%Create a bottom line array for use in filling plots, needs to be same length as N
bottomline2(2:N)=-20000000; %arbitrary very negative number

        
%% Run
for i=1:1:imax
    
    % Determine ELA
    ELA = startELA + (ampELA*sin(2*pi*t(i)/pELA)); % varying ELA with time around a starting ELA value
    ELAx = ones(length(xplot))*ELA; %make an array the size of x to fill with the ELA this is for plotting a moving line of ELA on glacier profile
    
    % Update Mass Balance
    bzero = gamma.*(iceelevation1 - startELA); %the initial condition for the b line, to be plotted in mass balance plot
    b = gamma.*(iceelevation1 - ELA); % recalculates b based on the updated ELA array for model 
    
    % Calculate Ice Flux 
    
    iceslope = diff(iceelevation1)./dx; % calculate slopes on ice surface   
    ticeslope = diff(ticeelevation1)./dx; % calculate slopes on ice surface   
    for j=1:1:jmax  %go through the x array
            edgeicethickness1(j) = (icethickness1(j)+icethickness1(j+1))/2; % update the ice thickness at the edge of each dx element 
    end
    for j=1:1:jmax  %go through the x array
            tedgeicethickness1(j) = (ticethickness1(j)+ticethickness1(j+1))/2; % update the ice thickness at the edge of each dx element 
    end
    
    Uslide = Uo.*exp(-edgeicethickness1./Hstar); %glacier sliding based on thickness of ice
    tUslide = Uo.*exp(-tedgeicethickness1./Hstar); %glacier sliding based on thickness of ice
    
    % Determine terminus positions
    mainterm = find(icethickness1==0,1); %find the terminus of the main glacier, useful info for checking model
    tribterm(i) = find(QflowTrib(2:N+1) == 0,1); %find the terminus of the tributary glacier, useful info for checking model
    
    % Determine Fluxes
    % Main Glacier Flux
        QflowMain(2:N) = (Uslide.*edgeicethickness1)-((1/valleywidth)*(A.*((rhoi*g*iceslope).^3).*((edgeicethickness1.^5)/5))); % Ice Flux at each position according to flow law
        QflowMain(1)=0; % no ice flux at the top, boundary condition
        QflowMain(N+1)=0; % no ice flux at the bottom, boundary condition
        
    % Tributary Glacier Flux 
        QflowTrib=QflowMain; %the tributary is an identical version of the main glacier, just rotated about the junction point, up until the junction
        %QflowTrib(junctionxarray+1) = -QflowTrib(junctionxarray); %after the junction, all ice is lost to main glacier instantly
        %QflowTrib(junctionxarray+1:N+1) = 0; %there are no fluxes beyond the junction area
        %QflowTrib(1)=0; %top boundary condition same as main glacier
   
    % Calculate Net Flux, dQdx, in Tributary
        tdQdx(1:junctionxarray) = diff(QflowTrib(1:junctionxarray+1)./dx); % calculate net flux through each dx element
        tdQdx(junctionxarray+1) = -tdQdx(junctionxarray);
        tdQdx(junctionxarray+2:N)=0;
    
    
    % Calculate Net Flux, dQdx, in Main Glacier without Tributary
        dQdx = diff(QflowMain)./dx; % calculate net flux through each dx element
    
    % Addition Array
        QaddTrib(junctionxarray) = tdQdx(junctionxarray); %in the addition array, insert the flux at the junction point from the main glacier in the junction position
 
    %Add the ice from the tributary to the main glacier
        if tdQdx(junctionxarray)<0
            dQdx(junctionxarray) = dQdx(junctionxarray)-tdQdx(junctionxarray+1); %add the addition array to the main glacier. nothing will be added if the tributary is not touching because the tributary flux at the junction point will be zero so zero is stored in the addition array at that timestep  
        end
    % Calculate Change in Ice Thickness   
        dhdt = b - dQdx; % change in ice thickness for time step is mass from b, minus the net flux via the flow law
        tdhdt = b - tdQdx; % change in ice thickness for time step is mass from b, minus the net flux via the flow law 
    % Update Total Ice Thickness in the Main Glacier
    icethickness1 = icethickness1 + (dhdt*dt); % update ice thickness with dhth
    Hneg = find(icethickness1<0); % don't let the ice thickness be negative
    icethickness1(Hneg)=0; % don't let the ice thickness be negative
    
    %Update Ice Thickness in Tributary
    ticethickness1 = ticethickness1 + (tdhdt*dt); % update ice thickness with dhth
    tHneg = find(ticethickness1<0); % don't let the ice thickness be negative
    ticethickness1(Hneg)=0; % don't let the ice thickness be negative
    
    
    % Calculate Erosion
    for k = 1:(length(icethickness1)-1) %go through the x space length
        if icethickness1(k) > 0 %find where there is ice
            erosionrate(k) = a*Uslide(k).^n; %if there is ice this is the erosion rate
            zrock1(k) = zrock1(k) - erosionrate(k)*dt; %lower the bedrock by the erosion rate and time step
        end
    end
    
    for k = 1:(length(ticethickness1)-1) %go through the x space length
        if ticethickness1(k) > 0 %find where there is ice
            terosionrate(k) = a*tUslide(k).^n; %if there is ice this is the erosion rate
            tzrock1(k) = tzrock1(k) - terosionrate(k)*dt; %lower the bedrock by the erosion rate and time step
        end
    end
  
    
    % Update Elevation of Ice in Prfile of Main Glacier
    iceelevation1 = zrock1 + icethickness1; % update ice elevation
    
     % Update Elevation of Ice in Prfile of Main Glacier
    ticeelevation1 = tzrock1 + ticethickness1; % update ice elevation
    

    
    % Make Plotting Easy
    iceplot(2:N+1)=iceelevation1; % for plotting the ice elevation
    thicknessplot(2:N+1)=icethickness1; % for plotting the ice thickness 

    
    % Plot results
    if rem(t(i),tplot)==0  
        figure(1)
        clf
        
        %Glacier Profile Subplot
            subplot('position',[.1 .45 .8 .5])
            plot(x,zrock1,'k','linewidth',1)
            hold all
            
            plot(xplot,iceplot,'c')
            
            %Color the Glacier
            xx = [xplot xplot];
            yy = [iceplot bottomline1];
            fill(xx,yy,[.62 .82 .882]);
            
            %Color the Bedrock
            xx = [x x];
            yy = [zrock1 bottomline2];
            fill(xx,yy,[.91,.89,.87]);
            
            %Add the ELA Line
            plot(xplot,ELAx,'Color',[.812 .49 .392], 'linewidth', 3)
            
            %Plot Formatting
            xlabel('Distance (m)', 'fontname', 'arial', 'fontsize',22)
            ylabel('Elevation (m)', 'fontname', 'arial', 'fontsize',22)
            title('Glacier with Ice Tributary')
            set(gca,'fontsize',18,'fontname','arial')
            ht=text(2500,6500,['  ',num2str(round(t(i))), ' yrs  '],'fontsize',20); %print time in animation
            axis([0 (2/3)*topomax 2500 topomax+1000])
            hold all 
            
        %Mass Balance Subplot
            subplot('position',[.1 .1 .2 .25])
            plot(b,iceelevation1,'Color',[.812 .49 .392], 'linewidth', 3)
            hold all
            
            %Plot Formatting
            plot(zerosforb,zplot, 'k--')
            plot(bzero,iceelevation1,'k')
            xlabel('Accumulation/Ablation (m/yr)', 'fontname', 'arial', 'fontsize',16)
            ylabel('Elevation (m)', 'fontname', 'arial', 'fontsize',16)
            set(gca,'fontsize',18,'fontname','arial')
            axis([-10 5 3500 topomax+10])
            hold all

        %Ice Thickness Subplot
            subplot('position',[.4 .1 .2 .25])
            fill(xplot,thicknessplot,[.62 .82 .882])
            hold all
            plot(junctionforthick,thickplot,'k--')
            
            %Plot Formatting
            xlabel('Distance (m)', 'fontname', 'arial', 'fontsize',16)
            ylabel('Ice Thickness (m)', 'fontname', 'arial', 'fontsize',16)
            set(gca,'fontsize',18,'fontname','arial')
            axis([0 (2/3)*topomax 0 400])
            hold all 

        %Ice Flux Subplot
            subplot('position',[.7 .1 .2 .25])
            plot(xplot2,QflowMain/10000,'Color',[.812 .392 .608], 'linewidth', 4)
            hold on
            
            %Plot Formatting
            xlabel('Distance (m)', 'fontname', 'arial', 'fontsize',16)
            ylabel('Ice Flux via Flow m^3/yr x 10^4', 'fontname', 'arial', 'fontsize',16)
            set(gca,'fontsize',18,'fontname','arial')
            axis([0 (2/3)*topomax 0 3])
    
        pause(0.1)
    end
end
       
        
        
        
        