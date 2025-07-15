function [simA,simGs] = simulate_ACI(Variable,KE_type)
% This function simulate ACi curves under changing Air CO2 (Ca) with constant PAR
%
% INPUTS:
%    Variable:              Kinetic parameters
%
%
% OUTPUT:
%    simA:                  Simulated photosynthetic rate
%    simGs:                 Simulated stomatal conductance

%% Leaf gas exchange measurements 
% Stabilization of A and gs at the beginning of measurements with 400 CO2 
% followed by stepwise increase until 1250
% When CO2 returned to 400 again, A and gs are allowed to restabilize

Ca_t = configC4Sim('Ca_t');

% Set empty vector for PAR changes
Q_t=[];

% Set experimental condition and load global variables for C4 model
air_RH=0.65;
air_temp=25;

% Load metabolite names
mets_name=load_metabolite_names();
[~,ind_gs]=ismember("Gs[Leaf]",mets_name);

% index of A in Gs_VEL
ind_A = 2;

% Load initial concentrations
envFactor=var_env(Q_t,Ca_t(1),air_temp,air_RH,[]);
Ini=RAC4leafMetaIni(envFactor);% Initial values
nm=length(Ini);


%% Start ACi record
% global Gs_VEL initialized in optim_initialization...
nchange=length(Ca_t);

% Initialize simulation vectors
simA=zeros(nchange,1);
simGs=zeros(nchange,1);

continue_flag=true;
i=1;
while continue_flag && i<(nchange+1)

    ode_sol=[];
    initialization_flux_output();
    global Gs_VEL
    if i==1
        xt0=Ini;
    else
        xt0=current_state;
    end

    if i==1 || i==6
        
        startTime = datetime('now');
        stop_fun = @(t,y)reaching_steadyA(t,y,startTime);
        tspan=[0 60*60]; % maximum 60 minutes for stabilization
        options=odeset('NonNegative',1:nm, 'RelTol', 1e-04,'Events',stop_fun); % stop event is steady state of A
    else
        startTime = datetime('now');
        stop_fun = @(t,y)timeLimit(t,y,startTime);
        tspan=[0 2*60]; % 120 seconds for stabilization
        options=odeset('NonNegative',1:nm, 'RelTol', 1e-04,'Events',stop_fun); 
    end

    envFactor=var_env(Q_t,Ca_t(i),air_temp,air_RH,[]);

    ode_sol=ode15s(@(t,x)RAC4leafMetaMB(t,x,Variable,KE_type,envFactor),tspan,xt0,options); 
    
    if i==1 || i==6
        if isempty(ode_sol.xe) || max(ode_sol.xe)==tspan(2) || Gs_VEL(end,ind_A)<1%if ode simulation fails or steady state is not reached by the end of 60min
            continue_flag=false;
        end
    else
        if isempty(ode_sol) || max(ode_sol.x)~=tspan(2)
            continue_flag=false;
        end
    end
    if continue_flag==true
        simA(i)=Gs_VEL(end,ind_A);
        simGs(i)=ode_sol.y(ind_gs,end);
        current_state=ode_sol.y(:,end);
    else
        simA(i)=1e5;
        simGs(i)=1e5;
    end
    i=i+1;
end

% The measurement at 6th point is not good according to John
simGs(6)=[];
simA(6)=[];

% simA=real(simA);
% simGs=real(simGs);




