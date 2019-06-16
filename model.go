
package main

import (

    "fmt"
    tg "github.com/galeone/tfgo"
    tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
    model := tg.LoadModel("deployment", []string{"serve"}, &tf.SessionOptions{})

    SepalLengthCm, _ := tf.NewTensor([]float64{4})
    SepalWidthCm, _ := tf.NewTensor([]float64{2})
    PetalLengthCm, _ := tf.NewTensor([]float64{1})
    PetalWidthCm, _ := tf.NewTensor([]float64{0.5})
    
    results := model.Exec(
        []tf.Output{
            model.Op("dnn/head/predictions/probabilities", 0),
        }, map[tf.Output]*tf.Tensor{
            model.Op("SepalLengthCm_placeholder", 0): SepalLengthCm,
            model.Op("SepalWidthCm_placeholder", 0):  SepalWidthCm,
            model.Op("PetalLengthCm_placeholder", 0): PetalLengthCm,
            model.Op("PetalWidthCm_placeholder", 0):  PetalWidthCm,

        },
    )
    predictions := results[0].Value().([][]float32)
    fmt.Println(predictions)
}