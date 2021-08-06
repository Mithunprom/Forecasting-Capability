----------------------------
----------------------------
----------------------------
--creating data for mendoza scenario

drop table if exists #DirectorVolumesTable;
drop table if exists zzz_director_allocation;
drop table if exists [zzz_Mendoza_actual];

select a.[VP]
      ,a.[DIRECTOR] 
      ,a.[PRODUCT_CATEGORY]
      ,a.[ORDER_ACTIVITY]
      ,left(a.[INSTALL_MO],4)*10000+right(a.[INSTALL_MO],2)*100+01 as DateInt
      ,sum(a.[UNITS]) as Director_Volumes  into #DirectorVolumesTable
       FROM [NAO_PMO_Analytics].[dbo].[zzz_Director_Input] a  ------------------------------------------ need excel
         where left(a.[INSTALL_MO],4)*10000+right(a.[INSTALL_MO],2)*100+01 between 20210101 and 20210401
       group by 
       a.[VP]
      ,a.[DIRECTOR] 
      ,a.[PRODUCT_CATEGORY]
      ,a.[ORDER_ACTIVITY]
      ,left(a.[INSTALL_MO],4)*10000+right(a.[INSTALL_MO],2)*100+01;

  select a.[VP]
      ,a.[DIRECTOR] 
      ,a.[PRODUCT_CATEGORY]
      ,a.[ORDER_ACTIVITY]
      ,sum(a.Director_Volumes) as Director_Volumes
      ,  sum(b.Volumes )as Product_Volumes
      ,sum(a.Director_Volumes)/sum(b.Volumes ) as Allocation_Percentage
	  into zzz_director_allocation
        FROM #DirectorVolumesTable a
         left join [NAO_PMO_Analytics].[dbo].[zzz_Forecast_Input] b
  on a.[PRODUCT_CATEGORY]=b.[Product] and a.[ORDER_ACTIVITY]=b.[Sub-Product]
  and a.DateInt=b.[Date]
  where a.DateInt between 20210101 and 20210401
  group by  a.[VP]
      ,a.[DIRECTOR] 
      ,a.[PRODUCT_CATEGORY]
      ,a.[ORDER_ACTIVITY];




SELECT CONVERT([Date], date) as Date
      ,[Product]
      ,[Subproduct]
      ,[Volumes]
into [NAO_PMO_Analytics].[dbo].[zzz_Mendoza_actual]
  FROM [NAO_PMO_Analytics].[dbo].[zzz_Mendoza_raw];------------------------------------------ need excel

select * from [NAO_PMO_Analytics].[dbo].[zzz_Mendoza_actual];

drop table if exists  [NAO_PMO_Analytics].[dbo].[zzz_final_unpvt_temp];

SELECT [Primary_key]
      ,t1.[date]
      ,t1.[Scenario]
      ,t1.[Product]
      ,t1.[Sub-Product]
      , ISNULL((case when exists (select Volumes from [NAO_PMO_Analytics].[dbo].[zzz_Mendoza_actual])
          then t2.Volumes  else t1.Volumes  end),0) as Volumes
      ,t1.[Methods]
      ,t1.[Forecasting]
	  into [zzz_final_unpvt_temp]
  FROM [NAO_PMO_Analytics].[dbo].[zzz_final_unpvt] t1 left join [NAO_PMO_Analytics].[dbo].[zzz_Mendoza_actual] t2 on
  t1.[Sub-Product]=t2.[Subproduct] and t1.[Product]=t2.Product and t1.Date=t2.Date

DROP TABLE IF EXISTS [NAO_PMO_Analytics].[dbo].zzz_dir_allocation01;
DROP TABLE IF EXISTS [NAO_PMO_Analytics].[dbo].zzz_table05_01;

SELECT [Name]
      ,REPLACE('Mendoza, Ronald (Ron)', [Group_Name], 'Mendoza') as Group_Name
      ,[Scenario]
      ,[Product]
      ,[SubProduct]
      ,[Weights]
      ,[Key]
      ,[F10]*100 as Allocation_Percentage
	  into zzz_dir_allocation01
  FROM [NAO_PMO_Analytics].[dbo].[zzz_dir_allocation]
  -----------------------------------------------
SELECT Distinct t1.[ID]
      ,t1.[Name]
      ,t1.[Group_Name]
      ,t1.[Scenario]
      ,t1.[Product]
      ,t1.[SubProduct]
      ,t1.[Weights]
      ,t2.[Allocation_Percentage]
	  into zzz_table05_01
  FROM [NAO_PMO_Analytics].[dbo].[zzz_tab05] t1 left join [NAO_PMO_Analytics].[dbo].zzz_dir_allocation01 t2
  on t1.[Name]=t2.Name and t1.[Group_Name] like t2.Group_name and t1.[Scenario]=t2.[Scenario] and t1.Product=t2.Product and t1.[SubProduct]=t2.[SubProduct]
 ;


---------------------------
---------------------------
---------------------------
--tables for the headcount modeling

DROP TABLE IF EXISTS [NAO_PMO_Analytics].[dbo].zzz_table_final_input;
SELECT t1.[Name]
      ,t1.[Group_Name]
      ,t1.[Scenario]
      ,t1.[Product]
      ,t1.[SubProduct]
      ,AVg([Weights]) as Weights
      ,Avg([Allocation_Percentage]) as Allocation_Percentage
      ,Avg([HeadCount]) as HeadCount
      ,Avg([UnitCost]) as UnitCost
      ,Avg([OperatingCost]) as OperatingCost
      ,Avg([Historical_HeadCount]) as Historical_HeadCount
	  ,replace(t3.[Methods],'Methods : ','') as Method
	  into [NAO_PMO_Analytics].[dbo].zzz_table_final_input
  FROM [NAO_PMO_Analytics].[dbo].zzz_table05_01 t1 left join [NAO_PMO_Analytics].[dbo].[zzz_tab06] t2
  on t1.Name=t2.Name and  t1.Group_Name=t2.[Group_name]  
  left join [NAO_PMO_Analytics].[dbo].[zzz_tab04] t3
  on t1.Name=t3.[Names] and t1.Scenario=replace(t3.[Scenario],'Scenario : ','') and t1.[Product]=replace(t3.[Product],'Product :','')
  and t1.[SubProduct]=replace(t3.[SubProduct],'Sub-product :','')
  group by t1.product, t1.SubProduct,t1.Group_name, t1.[Scenario],t1.[Name],t3.[Methods]

  --select * from [NAO_PMO_Analytics].[dbo].zzz_table_final_input where group_name='Mendoza' and name='Christopher Wang';

------------------------------------------------------------
DROP TABLE IF EXISTS [NAO_PMO_Analytics].[dbo].[zzz_table_final_01];
select a.[date]
,c.[Group_name]
,a.[Product]
,a.[Sub-Product]
,avg(Forecasting*Weights) as Weighted_Forecasting
,avg(Forecasting*Allocation_Percentage/100) as Allocated_Forecasting
,avg(Volumes*Weights) as Weighted_Actual
--,avg(Volumes*Allocation_Percentage/100) as Allocated_Actual
,avg(Volumes) as Allocated_Actual
,avg(Forecasting*Weights*Allocation_Percentage/100) as Weighted_allocated_Forecasting
--,avg(Volumes*Weights*Allocation_Percentage/100) as Weighted_allocated_Actual
,avg(Volumes*Weights) as Weighted_allocated_Actual
into [NAO_PMO_Analytics].[dbo].[zzz_table_final_01]
from [NAO_PMO_Analytics].[dbo].[zzz_final_unpvt_temp] a
join [NAO_PMO_Analytics].[dbo].[zzz_tab04] b
on  a.Methods=replace(b.[Methods],'Methods : ','')
and a.[Product]=replace(b.[Product],'Product :','')
and a.[Sub-Product]=replace(b.[SubProduct],'Sub-product :','')
and a.[Scenario]=replace(b.[Scenario],'Scenario : ','')
join [NAO_PMO_Analytics].[dbo].zzz_table05_01 c
on a.[Scenario]=c.[Scenario]
and a.[Product]=c.[Product]
and a.[Sub-Product]=c.[SubProduct]
where date>='2021-01-01'-------------------------------------------- if you want to change the date
group by a.[Product], a.[Sub-Product],c.[Group_name],a.[date],c.[Group_name] ;

--select * date, Sum(Allocated_Actual) from [NAO_PMO_Analytics].[dbo].[zzz_table_final_01] where group_name='Mendoza' group by date ;
-------------------------------------------------------------


DROP TABLE IF EXISTS [NAO_PMO_Analytics].[dbo].zzz_table_final_011;
  Select [date], t1.[Name] , t2.[Group_Name],t1.Scenario,t2.[Product]
      ,t2.[Sub-Product]
      ,[Weights]
      ,[Allocation_Percentage]
      ,[HeadCount]
      ,[UnitCost]
      ,[OperatingCost]
      ,[Historical_HeadCount]
      ,t1.[Method]
	  ,[Weighted_Forecasting]
      ,[Allocated_Forecasting]
      ,[Weighted_Actual]
      ,[Allocated_Actual]
      ,[Weighted_allocated_Forecasting]
      ,[Weighted_allocated_Actual]
	   into zzz_table_final_011
	   FROM [NAO_PMO_Analytics].[dbo].[zzz_table_final_input] t1 join
	   [NAO_PMO_Analytics].[dbo].[zzz_table_final_01] t2 on
	   t2.Group_name=t1.Group_Name and t2.Product=t1.Product and t2.[Sub-Product]=t1.SubProduct
	   where date>='2021-01-01'-------------------------------------------- if you want to change the date
	   order by date;

	   
select * from [NAO_PMO_Analytics].[dbo].zzz_table_final_011 where group_name='Mendoza' and date='2021-01-01';

--------------------------------------------------------------
DROP TABLE IF EXISTS [NAO_PMO_Analytics].[dbo].[zzz_table_final_022];

select a.[date],a.[Group_Name]
      ,SUM([Weighted_allocated_Forecasting]*[UnitCost]/([OperatingCost]/[HeadCount])) as forecasted_Headcount
	  ,[Historical_HeadCount] into [NAO_PMO_Analytics].[dbo].[zzz_table_final_022]
from [NAO_PMO_Analytics].[dbo].[zzz_table_final_01] a
join [NAO_PMO_Analytics].[dbo].[zzz_tab06] b
on a.[Group_name]=b.[group_name]
where date>='2021-01-01' -------------------------------------------- if you want to change the date
group by a.[Group_name],a.[date],[Historical_HeadCount];

--select * from [NAO_PMO_Analytics].[dbo].[zzz_table_final_022] where group_name='Mendoza'and date='2021-01-01';

