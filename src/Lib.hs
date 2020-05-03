-- Logistic Regression module
module Lib
  ( model
  ) where

import Data.Matrix
import qualified Data.Vector as V

-- X
type Features = Matrix Double

-- Y
type Outputs = Matrix Double

-- A
type Activations = Matrix Double

-- W
type Weights = Matrix Double

-- dW
type WeightsDerivatives = Matrix Double

-- b
type Bias = Double

-- db
type BiasDerivative = Double

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
answers :: Features -> Weights -> Bias -> Activations
answers xss ws b = activation . (+ b) <$> matrixProduct (transpose ws) xss

-- loss a y = -y * log a - (1 - y) * log (1 - a)
-- j = mean [loss a y]
-- dJ / dw = 1/m X * (A - Y).T
-- where * is a matrix product
-- xss - dataset of (feature number, example number) shape
-- as - dataset of (1, example number) shape
-- ys - dataset of (1, example number) shape
weightDerivations :: Features -> Activations -> Outputs -> WeightsDerivatives
weightDerivations xss as ys =
  fmap (/ (fromIntegral $ ncols xss)) $
  matrixProduct xss $
  transpose $ fromList 1 (ncols xss) $ zipWith (-) (toList as) (toList ys)

-- dJ / db = mean (a - y)
-- as - dataset of (1, example number) shape
-- ys - dataset of (1, example number) shape
biasDerivation :: Activations -> Outputs -> BiasDerivative
biasDerivation as ys = mean $ V.zipWith (-) (getRow 1 as) (getRow 1 ys)

forPropagation :: Features -> Weights -> Bias -> Activations
forPropagation xss ws b = as
  where
    as = answers xss ws b

backPropagation ::
     Features -> Activations -> Outputs -> (WeightsDerivatives, BiasDerivative)
backPropagation xss as ys = (dws, db)
  where
    dws = weightDerivations xss as ys
    db = biasDerivation as ys

propagation ::
     Features
  -> Weights
  -> Bias
  -> Outputs
  -> (WeightsDerivatives, BiasDerivative)
propagation xss ws b ys = (dws, db)
  where
    as = forPropagation xss ws b
    (dws, db) = backPropagation xss as ys

-- -> (ws, b, dws, db, iterations)
descent ::
     Features
  -> Weights
  -> Bias
  -> Outputs
  -> Double
  -> Int
  -> Int
  -> (Weights, Bias, WeightsDerivatives, BiasDerivative)
descent xss ws b ys rate iterations iteration =
  if iteration >= iterations
    then (newWs, newB, dws, db)
    else descent xss newWs newB ys rate iterations (iteration + 1)
  where
    (dws, db) = propagation xss ws b ys
    newWs =
      fromList (nrows ws) (ncols ws) $
      zipWith (-) (toList ws) ((* rate) <$> toList dws)
    newB = b - rate * db

prediction :: Matrix Double -> Matrix Double -> Double -> Matrix Double
prediction xss ws b =
  (\x ->
     if x >= 0.5
       then 1
       else 0) <$>
  answers xss ws b

model ::
     Features
  -> Outputs
  -> Features
  -> Outputs
  -> Int
  -> Double
  -> (Outputs, Outputs, Weights, Bias)
model xssTrain ysTrain xssTest ysTest iterations rate =
  (yTestPrediction, yTrainPrediction, ws, b)
  where
    yTestPrediction = prediction xssTest ws b
    yTrainPrediction = prediction xssTrain ws b
    initWs =
      matrix
        (V.length $ getCol 1 xssTrain)
        1
        (\(i, j) -> fromIntegral (i + j) * 0.01)
    initB = 0
    (ws, b, dw, db) = descent xssTrain initWs initB ysTrain rate iterations 0
