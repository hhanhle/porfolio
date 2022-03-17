
-- Show sale data table & order by date

SELECT * 
   FROM [Sample].[dbo].[AdventureWorks_Sales_2015]
   ORDER BY 1, 2

-- Union Sale 2015 - 2017
SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2015]
UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2016])
UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2017])


-- Create table of Union Sale 2015 - 2017
SELECT * INTO Comprehensive_sales FROM
(SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2015]
UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2016])
UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2017])
)

--Create sales calculation table
SELECT b.*, [OrderDate],[StockDate],[OrderNumber],[CustomerKey],[TerritoryKey],[OrderLineItem],[OrderQuantity], TotalSales = cs.OrderQuantity * b.ProductPrice INTO Sales_report
   FROM Comprehensive_sales cs
JOIN [Sample].[dbo].[AdventureWorks_Products] b
   ON cs.ProductKey = b.ProductKey

-- Looking at number of order by productkey

SELECT Sales.ProductKey, Sales.TerritoryKey, SUM(Sales.OrderQuantity) as Order_quantity
   FROM (
	     SELECT * 
            FROM [Sample].[dbo].[AdventureWorks_Sales_2015]
         UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2016])
         UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2017])) as Sales
GROUP BY Sales.ProductKey, Sales.TerritoryKey

--Join return_table to get return quantity

SELECT sa.OrderDate, sa.StockDate, sa.OrderQuantity, sa.ProductKey, sa.TerritoryKey, re.ReturnQuantity, re.ReturnDate
   FROM (
	     SELECT * 
            FROM [Sample].[dbo].[AdventureWorks_Sales_2015]
         UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2016])) as sa
LEFT JOIN [Sample].[dbo].[AdventureWorks_Returns] re
   ON sa.ProductKey = re.ProductKey AND sa.TerritoryKey = re.TerritoryKey
WHERE sa.OrderDate < re.ReturnDate


-- Calculate return rate by productkey

With Order_amount as (
SELECT Sales.ProductKey, Sales.TerritoryKey, Sales.OrderDate, SUM(Sales.OrderQuantity) as Order_quantity
   FROM (
	     SELECT * 
            FROM [Sample].[dbo].[AdventureWorks_Sales_2015]
         UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2016])
         UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2017])) as Sales
GROUP BY Sales.ProductKey, Sales.TerritoryKey, Sales.OrderDate),
Return_amount as (
SELECT ProductKey, TerritoryKey, ReturnDate, SUM(ReturnQuantity) as Return_quantity
   FROM [Sample].[dbo].[AdventureWorks_Returns]
GROUP BY ProductKey, TerritoryKey, ReturnDate),
Product_summary as (
SELECT a.*, ReturnDate, Return_quantity
   FROM Order_amount a
LEFT JOIN Return_amount b
ON a.ProductKey = b.ProductKey AND a.TerritoryKey = b.TerritoryKey)

SELECT  ps.OrderDate, ps.ReturnDate, Date = ISNULL(ps.ReturnDate, ps.OrderDate), ps.ProductKey, ps.Order_quantity, ps.Return_quantity,  
        Return_rate = convert(decimal(5,2), ISNULL(Return_quantity * 1.00/Order_quantity,0)),		
		p.ProductSKU, p.ProductName, p.ModelName, p.ProductCost, p.ProductPrice, TerritoryKey 
   FROM Product_summary ps
LEFT JOIN [Sample].[dbo].[AdventureWorks_Products] p
ON ps.ProductKey = p.ProductKey
ORDER BY 4 desc


--Create Product return rate table
With Order_amount as (
SELECT Sales.ProductKey, Sales.TerritoryKey, Sales.OrderDate, SUM(Sales.OrderQuantity) as Order_quantity
   FROM (
	     SELECT * 
            FROM [Sample].[dbo].[AdventureWorks_Sales_2015]
         UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2016])
         UNION (SELECT * FROM [Sample].[dbo].[AdventureWorks_Sales_2017])) as Sales
GROUP BY Sales.ProductKey, Sales.TerritoryKey, Sales.OrderDate),
Return_amount as (
SELECT ProductKey, TerritoryKey, ReturnDate, SUM(ReturnQuantity) as Return_quantity
   FROM [Sample].[dbo].[AdventureWorks_Returns]
GROUP BY ProductKey, TerritoryKey, ReturnDate),
Product_summary as (
SELECT a.*, ReturnDate, Return_quantity
   FROM Order_amount a
LEFT JOIN Return_amount b
ON a.ProductKey = b.ProductKey AND a.TerritoryKey = b.TerritoryKey)

SELECT  ps.OrderDate, ps.ReturnDate, Date = ISNULL(ps.ReturnDate, ps.OrderDate), ps.ProductKey, ps.Order_quantity, ps.Return_quantity,  
        Return_rate = convert(decimal(5,2), ISNULL(Return_quantity * 1.00/Order_quantity,0)),		
		p.ProductSKU, p.ProductName, p.ModelName, p.ProductCost, p.ProductPrice, TerritoryKey INTO Product_return_summary_table
   FROM Product_summary ps
LEFT JOIN [Sample].[dbo].[AdventureWorks_Products] p
ON ps.ProductKey = p.ProductKey
ORDER BY 4 desc





