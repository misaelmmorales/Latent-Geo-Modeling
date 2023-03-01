import sgems

sgems.execute('DeleteObjects SIM')
sgems.execute('DeleteObjects finished')
sgems.execute('NewCartesianGrid  SIM::128::128::16::1::1::1::0::0::0')
sgems.execute('LoadObjectFromFile snesim_std.ti::All')
sgems.execute('DeleteObjects finished')
sgems.execute('RunGeostatAlgorithm  snesim_std::/GeostatParamUtils/XML::<parameters ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" ="" Nb_Multigrids_ADVANCED="5" ="" ="" ="" ="" ="" ="" ="" ="" ="" Scan_Template="11 11 11" Data_Weights="0.4 0.4 0.2">    <algorithm name="snesim_std" />    <GridSelector_Sim value="SIM" />    <Property_Name_Sim value="snesim_std" />    <Nb_Realizations value="6" />    <Seed value="211175" />    <PropertySelector_Training grid="TI" property="facies" />    <Nb_Facies value="2" />    <Marginal_Cdf value="0.723312 0.276688" />    <Max_Cond value="60" />    <Search_Ellipsoid value="80 80 80 0 0 0" />    <Hard_Data grid="" property="" />    <Use_ProbField value="0" />    <ProbField_properties count="0" value="" />    <TauModelObject value="1 1" />    <VerticalPropObject value="" />    <VerticalProperties count="0" value="" />    <Use_Affinity value="0" />    <Use_Rotation value="0" />    <Cmin value="1" />    <Constraint_Marginal_ADVANCED value="0.5" />    <resimulation_criterion value="-1" />    <resimulation_iteration_nb value="1" />    <Debug_Level value="0" />    <Subgrid_choice value="0" />    <expand_isotropic value="1" />    <expand_anisotropic value="0" />    <aniso_factor value="" />    <Region_Indicator_Prop value="snesim_std q__real0" />    <Active_Region_Code value="" />    <Use_Previous_Simulation value="0" />    <Use_Region value="0" /></parameters>')


sgems.execute('SaveGeostatGrid  SIM::snesim_std.out::gslib::0::snesim_std__real0::snesim_std__real1::snesim_std__real2::snesim_std__real3::snesim_std__real4::snesim_std__real5')
sgems.execute('SaveGeostatGrid  SIM::snesim_std.sgems::s-gems::0::snesim_std__real0::snesim_std__real1::snesim_std__real2::snesim_std__real3::snesim_std__real4::snesim_std__real5')


sgems.execute('NewCartesianGrid  finished::1::1::1::1.0::1.0::1.0::0::0::0')
data=[]
data.append(1)
sgems.set_property('finished','dummy',data)
sgems.execute('SaveGeostatGrid  finished::finished::gslib::0::dummy')
