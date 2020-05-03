module Main where

import Data.Matrix
import Lib

getOr :: (Features, Outputs)
getOr = (xss, ys)
  where
    xMatrix = transpose $ fromLists [[0, 0], [0, 1], [1, 0], [1, 1]]
    yMatrix = fromLists [map (foldl numOr 0) $ toLists $ transpose xMatrix]
    xss = Features xMatrix
    ys = Outputs yMatrix

numOr :: (Eq a1, Eq a2, Num a1, Num a2, Num p) => a1 -> a2 -> p
numOr a b =
  if a /= 0 || b /= 0
    then 1
    else 0

main :: IO ()
main = do
  putStrLn "Test is done using function `or` with following data:"
  let xss = fst getOr
  let ys = snd getOr
  putStrLn "X:"
  putStrLn $ prettyMatrix $ xToMatrix xss
  putStrLn "Y:"
  putStrLn $ prettyMatrix $ yToMatrix ys
  let iterations = 1000
  let rate = 0.05
  let (predictions, _, ws, b) = model xss ys xss ys iterations rate
  putStrLn $
    "Predictions using " ++
    show iterations ++ " iterations with learning rate " ++ show rate ++ ":"
  putStrLn $ prettyMatrix $ yToMatrix predictions
  putStrLn "Weights:"
  putStrLn $ prettyMatrix $ wToMatrix ws
  putStrLn $ "Bias: " ++ show (bToDouble b)
