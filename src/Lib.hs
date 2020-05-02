-- Logistic Regression module
module Lib
  ( model
  ) where

import Data.Matrix
import qualified Data.Vector as V

-- xs - dataset of (featur number, 1) shape
-- ws - dataset of (featur number, 1) shape
scalarProduct :: Num a => Matrix a -> Matrix a -> a
scalarProduct xs ws = sum $ V.zipWith (*) (getCol 1 xs) (getCol 1 ws)

matrixProduct :: Num a => Matrix a -> Matrix a -> Matrix a
matrixProduct = multStd2

mean :: (Fractional a, Foldable t) => t a -> a
mean xs = sum xs / fromIntegral (length xs)

activation :: (Floating a) => a -> a
activation = sigmoid

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + e)
  where
    e = exp $ -x

-- as = g $ ws.T * xss + b
-- as - dataset of (1, example number) shape
-- xss - dataset of (feature number, example number) shape
-- ws - dataset of (feature number, 1) shape
answers :: Matrix Double -> Matrix Double -> Double -> Matrix Double
answers xss ws b = activation . (+ b) <$> matrixProduct (transpose ws) xss

loss :: Floating a => a -> a -> a
loss a y = -y * log a - (1 - y) * log (1 - a)

-- j = mean [loss a y]
-- as - dataset of (1, example number) shape
-- ys - dataset of (1, example number) shape
cost :: Floating a => Matrix a -> Matrix a -> a
cost as ys = mean $ V.zipWith loss (getRow 1 as) (getRow 1 ys)

-- dJ / dw = 1/m X * (A - Y).T
-- where * is a matrix product
-- xss - dataset of (feature number, example number) shape
-- as - dataset of (1, example number) shape
-- ys - dataset of (1, example number) shape
weightDerivation :: Matrix Double -> Matrix Double -> Matrix Double -> Double
weightDerivation xss as ys =
  mean $
  matrixProduct xss $
  transpose $
  fromList 1 (V.length $ getRow 1 xss) $ zipWith (-) (toList as) (toList ys)

-- dJ / db = mean (a - y)
-- as - dataset of (1, example number) shape
-- ys - dataset of (1, example number) shape
-- biasDerivation :: Floating a => V.Vector a -> V.Vector a -> a
biasDerivation :: Floating a => Matrix a -> Matrix a -> a
biasDerivation as ys = mean $ V.zipWith (-) (getRow 1 as) (getRow 1 ys)

forPropagation ::
     Matrix Double -> Double -> Matrix Double -> Matrix Double -> Matrix Double
forPropagation ws b xss ys = as
  where
    as = answers xss ws b
    costVal = cost as ys

backPropagation ::
     Matrix Double -> Matrix Double -> Matrix Double -> (Double, Double)
backPropagation xss as ys = (dw, db)
  where
    dw = weightDerivation xss as ys
    db = biasDerivation as ys

propagation ::
     Matrix Double
  -> Double
  -> Matrix Double
  -> Matrix Double
  -> (Double, Double)
propagation ws b xss ys = (dw, db)
  where
    as = forPropagation ws b xss ys
    (dw, db) = backPropagation xss as ys

-- -> (ws, b, dw, db)
descent ::
     Matrix Double
  -> Double
  -> Matrix Double
  -> Matrix Double
  -> Double
  -> Int
  -> Int
  -> (Matrix Double, Double, Double, Double, Int)
descent ws b xss ys rate iterations iteration =
  if iteration >= iterations
    then (newWs, newB, dw, db, iteration)
    else descent newWs newB xss ys rate iterations (iteration + 1)
  where
    (dw, db) = propagation ws b xss ys
    newWs = (-) (rate * dw) <$> ws
    newB = b - rate * db

prediction :: Matrix Double -> Matrix Double -> Double -> Matrix Double
prediction xss ws b =
  (\x ->
     if x >= 0.5
       then 1
       else 0) <$>
  answers xss ws b

model ::
     Matrix Double
  -> Matrix Double
  -> Matrix Double
  -> Matrix Double
  -> Int
  -> Double
  -> (Matrix Double, Matrix Double, Matrix Double, Double, Int)
model xssTrain ysTrain xssTest ysTest iterations rate =
  (yTestPrediction, yTrainPrediction, ws, b, iters)
  where
    yTestPrediction = prediction xssTest ws b
    yTrainPrediction = prediction xssTrain ws b
    initWs = matrix (V.length $ getCol 1 xssTrain) 1 $ const 0
    initB = 0
    (ws, b, dw, db, iters) =
      descent initWs initB xssTrain ysTrain rate iterations 0

getXor :: (Matrix Double, Matrix Double)
getXor =
  ( transpose $ fromLists [[0, 0], [0, 1], [1, 0], [1, 1]]
  , fromLists [[0, 1, 1, 0]])
