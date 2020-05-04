module Lib
  ( model
  , Features(..)
  , Outputs(..)
  , Weights(..)
  , Bias(..)
  ) where

import Data.Matrix
import qualified Data.Vector as V

-- X
newtype Features =
  Features (Matrix Double)

-- Y
newtype Outputs =
  Outputs (Matrix Double)

-- A
newtype Activations =
  Activations (Matrix Double)

-- W
newtype Weights =
  Weights (Matrix Double)

-- dW
newtype WeightsDerivatives =
  WeightsDerivatives (Matrix Double)

-- b
newtype Bias =
  Bias Double

-- db
newtype BiasDerivative =
  BiasDerivative Double

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
answers xss ws b =
  Activations $
  activation . (+ bDouble) <$> matrixProduct (transpose wsMatrix) xMatrix
  where
    Features xMatrix = xss
    Bias bDouble = b
    Weights wsMatrix = ws

-- loss a y = -y * log a - (1 - y) * log (1 - a)
-- j = mean [loss a y]
-- dJ / dw = 1/m X * (A - Y).T
-- where * is a matrix product
-- xss - dataset of (feature number, example number) shape
-- as - dataset of (1, example number) shape
-- ys - dataset of (1, example number) shape
weightsDerivatives :: Features -> Activations -> Outputs -> WeightsDerivatives
weightsDerivatives xss as ys =
  WeightsDerivatives $
  fmap (/ (fromIntegral $ ncols xMatrix)) $
  matrixProduct xMatrix $
  transpose $
  fromList 1 (ncols xMatrix) $ zipWith (-) (toList asList) (toList ysList)
  where
    Features xMatrix = xss
    Activations asList = as
    Outputs ysList = ys

-- dJ / db = mean (a - y)
-- as - dataset of (1, example number) shape
-- ys - dataset of (1, example number) shape
biasDerivative :: Activations -> Outputs -> BiasDerivative
biasDerivative as ys =
  BiasDerivative $ mean $ V.zipWith (-) (getRow 1 asMatrix) (getRow 1 ysMatrix)
  where
    Activations asMatrix = as
    Outputs ysMatrix = ys

forPropagation :: Features -> Weights -> Bias -> Activations
forPropagation xss ws b = as
  where
    as = answers xss ws b

backPropagation ::
     Features -> Activations -> Outputs -> (WeightsDerivatives, BiasDerivative)
backPropagation xss as ys = (dws, db)
  where
    dws = weightsDerivatives xss as ys
    db = biasDerivative as ys

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
      Weights $
      fromList (nrows wsMatrix) (ncols wsMatrix) $
      zipWith (-) (toList wsMatrix) ((* rate) <$> toList dwsMatrix)
    newB = Bias $ bDouble - rate * dbDouble
    Weights wsMatrix = ws
    WeightsDerivatives dwsMatrix = dws
    Bias bDouble = b
    BiasDerivative dbDouble = db

prediction :: Features -> Weights -> Bias -> Outputs
prediction xss ws b =
  Outputs $
  (\x ->
     if x >= 0.5
       then 1
       else 0) <$>
  aMatrix
  where
    Activations aMatrix = answers xss ws b

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
      Weights $
      matrix
        (V.length $ getCol 1 xTrainMatrix)
        1
        (\(i, j) -> fromIntegral (i + j) * 0.01)
    initB = Bias 0
    (ws, b, dw, db) = descent xssTrain initWs initB ysTrain rate iterations 0
    Features xTrainMatrix = xssTrain
