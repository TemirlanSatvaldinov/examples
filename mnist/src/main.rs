
use std::fs::File;
use std::io::{BufReader, BufRead, Write};
use rand::seq::SliceRandom;
use rand_distr::{Normal,Distribution};
use rand::{thread_rng};

use aicore;
use aicore::matrix::mul::matrix_mul;
use aicore::matrix::shape::transpose_matrix;
use aicore::function::activation::{sigmoid,sigmoid_derivative,softmax};
use aicore::function::loss::{cross_entropy};

const BUFFER_SIZE: usize = 2000;
 #[derive(Clone)]
pub struct Layer {
    pub n:usize,
    pub channel:usize,
    pub hight:usize,
    pub width:usize,
    pub weight:Vec<f32>
}
impl Layer {
    pub fn new_with_normalization(n:usize,channel:usize,high:usize,width:usize,mean:f32,max:f32)  -> Self{
        let weight_len:usize = n * channel * high * width;
        let mut   weight:Vec<f32> = vec![0.0; weight_len];
        let mut rng = thread_rng();
        let normal = Normal::new(mean,max).unwrap();
        for i in 0..weight_len {
           weight[i] = normal.sample(&mut rng); 
             
        }

       Layer { n, channel, hight: high, width, weight: weight }
    }
   
    
}


fn main () {
    
    let train_data_path = "../mnist_csv/mnist_train_100.csv";
    let test_data_path = "../mnist_csv/mnist_test_10.csv";
    
    

    let mut w1 = Layer::new_with_normalization(1, 1, 200, 784, 0.0, f32::powf(784.0,-0.5));
    // o1 =  w1 @ input = 200x784 @ 784x1
    let mut o1 = vec![0.0f32;200 * 1];
    let mut dl_t1 = vec![0.0f32;o1.len()];
    let mut dl_w1 = vec![0.0f32;w1.weight.len()];

    let mut w2 = Layer::new_with_normalization(1, 1, 10, 200, 0.0, f32::powf(200.0,-0.5));
    // o2 = w2 @ o1 = 10x200 @ 200x1
    let mut o2 = vec![0.0f32;10 * 1];
    let mut dl_w2 = vec![0.0f32;w2.weight.len()];
    // errs 
    let mut net_error = vec![0.0f32; 10*1];
    let mut hidden_error = vec![0.0f32;o1.len()];
    let mut transpose_buf = vec![0.0f32; w1.weight.len()];
   

    let mut rng = thread_rng();
    let epoch =  10;
    let lr_rate = 0.03;

    // train
    let packet_size = 100;
    let mut train_data:Vec<(Vec<f32>,Vec<f32>)> =  vec![(vec![0.01f32;10],vec![0.0f32;784]);packet_size];
    img_from_csv(train_data_path, &mut  train_data,packet_size).ok();
    for _ in 0..epoch {
       
        // shuffle data
        //train_data.shuffle(&mut rng);
        for image in 0..packet_size  {
            
            // forward
            matrix_mul(&w1.weight,w1.hight, w1.width, &train_data[image].1, 784, 1,&mut o1);
            o1.iter_mut().for_each(|x| *x = sigmoid(*x));
            
            matrix_mul(&w2.weight,w2.hight, w2.width, &o1, 200, 1,&mut o2);
            softmax(&mut o2);

            /* backward */ 
            //  neural network error = target -predict
            for i in 0..o2.len() {
                net_error[i] =  train_data[image].0[i] - o2[i];
            }
            // calculate hidden layer  err  = w2.T @ network_error
            transpose_matrix(&w2.weight, w2.hight, w2.width, &mut transpose_buf);
            matrix_mul(&transpose_buf, w2.width, w2.hight, &net_error,10, 1, &mut hidden_error);

            // new w2 = w2 + prev_err @ input.T * lr_rate
            transpose_matrix(&o1, 200, 1, &mut transpose_buf);
            matrix_mul(&net_error, 10, 1, &transpose_buf, 1, 200, &mut dl_w2);     
            aicore::sgd::weight::sgd_update(&mut w2.weight, &dl_w2, lr_rate, 0.0, 1.0);

            // new w1 = w1 + (hidden_err * activation derivative) @ input.T * lr_rate
            transpose_matrix(&train_data[image].1, 784, 1, &mut transpose_buf);
            for i in 0..dl_t1.len() {
                dl_t1[i] = sigmoid_derivative(hidden_error[i]);
            }
            matrix_mul(&dl_t1, 200, 1,&transpose_buf , 1, 784, &mut dl_w1);
            aicore::sgd::weight::sgd_update(&mut w1.weight, &dl_w1, lr_rate, 0.0, 1.0);

        }
    }
    let packet_size = 10;
    let mut test_data:Vec<(Vec<f32>,Vec<f32>)> =  vec![(vec![0.01f32;10],vec![0.0f32;784]);packet_size];
    img_from_csv(test_data_path, &mut  test_data,packet_size).ok();
    for image in 0..10 {

        matrix_mul(&w1.weight,w1.hight, w1.width, &test_data[image].1, 784, 1,&mut o1);
        o1.iter_mut().for_each(|x| *x = sigmoid(*x));
        
        matrix_mul(&w2.weight,w2.hight, w2.width, &o1, 200, 1,&mut o2);
        softmax(&mut o2);

        let loss = cross_entropy(&test_data[image].0, &o2);
        let (t,p )= get_max_arg(&test_data[image].0, &o2);
        println!("iter = {}, target = {}, predict = {}, loss = {}",image,t,p,loss);
    }
    // query
}

// res = vec(lbl,img)
fn img_from_csv (path:&str,images:&mut Vec<(Vec<f32>,Vec<f32>)>,packet_size:usize) -> Result<(), Box<dyn std::error::Error>> {
    if packet_size < 1 {
        return Ok(());
    }
    let file = File::open(path);
    if file.is_err() {
        panic!("cannot open file")
    }
    let  reader = BufReader::with_capacity(BUFFER_SIZE, file.unwrap());
    let mut index_image = 0;
    let mut index_inner = 0;

    for line in reader.lines() {
        match line {
            
            Ok(line) => {
                let lbl_raw = line[..1].parse::<f32>();
                if lbl_raw.is_err() {
                    panic!("cannot cast lbl to f32")
                }

                let i = *lbl_raw.as_ref(). unwrap() as usize;
                images[index_image].0[i] =  0.99;
                let values  = line[2..].split(",");
                for  v in values.into_iter() {
                    let x  = v.parse::<f32>();
                    if x.is_err() {
                        panic!("cannot cast str to f32")
                    }
                    images[index_image].1[index_inner] = (x.unwrap() / 255.0 * 0.99) + 0.01;
                    index_inner+=1;
                    }
                },
            Err(err) => panic!("cannot read line"),
        }
            
        if (packet_size-1) == index_image  {
            return Ok(())
        }
        index_inner = 0;
        index_image +=1;
    }
    Ok(())
}
fn get_max_arg (target:&Vec<f32>,pred :&Vec<f32>)-> (usize,usize) {
    let mut max = -std::f32::INFINITY;
    let mut maxb = -std::f32::INFINITY;
    let mut target_index = 0;
    let mut predict_index = 0;
    for i in 0..target.len() {
         if target[i] > max {
             max = target[i];
             target_index = i;
         }
         if pred[i] > maxb {
             maxb = pred[i];
             predict_index = i;
         }
    } 
    return  (target_index,predict_index)
 }