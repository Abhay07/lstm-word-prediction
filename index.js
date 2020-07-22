const tf = require('@tensorflow/tfjs-node-gpu');
const tokenizer = require('tokenizer.js');

const text = `It was during his interactions in the PWA Sunday meetings that Shailendra and Bimal Roy encouraged him to join films. Gulzar began his career under film directors Bimal Roy and Hrishikesh Mukherjee. His book Ravi Paar has a narrative of Bimal Roy and the agony of creation. He started his career as a songwriter with the music director Sachin Dev Burman for the movie Bandini (1963). In films, he found an environment associated with literature in the group he worked with, including Bimal Roy, most of whose films were based on literary works.[9] Shailendra, who has penned the rest of the songs in the movie requested Gulzar to write the song "Mora Gora Ang Layle", sung by Lata Mangeshkar.[3][4][10]

Directed and produced by Hrishikesh Mukherjee, the 1968 film Aashirwad had dialogues and lyrics written by Gulzar. Song lyrics and poems written by Gulzar gave the poetic attribute and the "much-needed additional dimension"[11] to Ashok Kumar's role in the film. Ashok Kumar received the Best Actor at the Filmfare and at the National Film Awards for this role.[11] Gulzar's lyrics, however, did not gain much attention until 1969's Khamoshi, where his song "Humne Dekhi Hai Un Aankhon Ki Mehekti Khushboo" (lit., "I have seen the fragrance of those eyes") became popular. Ganesh Anantharaman in his book Bollywood Melodies describes Gulzar's lyrics, with the purposeful mixing of the senses, to be "daringly defiant".[12][a][13] For the 1971 film Guddi, he penned two songs, of which "Humko Man Ki Shakti Dena" was a prayer which is still sung in many schools in India.[14]

As a lyricist, Gulzar had a close association with the music director Rahul Dev Burman. He has also worked with Sachin Dev Burman, Shankar Jaikishan, Hemant Kumar, Laxmikant-Pyarelal, Madan Mohan, Rajesh Roshan, and Anu Malik.[3][4][10][15] Gulzar worked with Salil Chowdhury in Anand (1971) and Mere Apne (1971); Madan Mohan in Mausam (1975), and more recently with Vishal Bhardwaj in Maachis (1996), Omkara (2006) and Kaminey (2009); A. R. Rahman in Dil Se.. (1998), Guru (2007), Slumdog Millionaire (2008) and Raavan (2010) and Shankar–Ehsaan–Loy in Bunty Aur Babli (2005).[3][4][10] Gulzar took inspiration from Amir Khusrow's "Ay Sarbathe Aashiqui" to pen "Ay Hairathe Aashiqui" for Mani Ratnam's 2007 Hindi film Guru, which had music composed by A. R. Rahman.[16] Another Ratnam-Rahman hit, "Chaiyya Chaiyya" from Dil Se.. also had lyrics written by Gulzar, based on the Sufi folk song "Thaiyya Thaiyya", with lyrics by poet Bulleh Shah.[17] For another collaboration with Rahman for Danny Boyle's 2007 Hollywood film Slumdog Millionaire, Rahman and Gulzar won the Academy Award for Best Original Song for "Jai Ho" at the 81st Academy Awards. The song received international acclaim and won him a Grammy Award (shared with Rahman) in the category of Grammy Award for Best Song Written for a Motion Picture, Television or Other Visual Media.[3][4][18][19] He also wrote a song for the Pakistani Drama Shehryar Shehzadi, and this song Teri Raza, has been sung by Rekha Bhardwaj and was composed by Vishal Bhardwaj.`

function pad_array_left(arr, len, fill) { 
	return Array(len).fill(fill).concat(arr).slice(arr.length); 
}

function createModel(){
	var model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: tokenizer.vocabSize, outputDim: 50 }))
    model.add(tf.layers.lstm({ units: 50 }));
    model.add(tf.layers.dense({ units: tokenizer.vocabSize, activation: 'sigmoid' }))
    model.summary();
    return model;
}

function convertToTensor(X, Y) {
    var inputs = tf.tensor(X);
    var labels = tf.tensor(Y);
    return {
        inputs,
        labels
    }
}

function oneHot(size, at) {
    var vector = [];
    for (var i = 0; i < size; i++) {
        if (at == i) {
            vector.push(1);
        } else {
            vector.push(0);
        }
    }
    return vector;
}

async function run() {
    text = text.toLowerCase();
    text = text.match(/[a-z\s.]/g).join("");
    var lines = text.split("\n");
    var tokenizer = new Tokenizer();
    tokenizer.fitOnTexts(lines);
    var inputSequences = [];
    var tokenList = tokenizer.textsToSequences(lines);
    //console.log(tokenList);
    var maxTokenLength = 0;
    for (var i in tokenList) {
        if (tokenList[i].length > maxTokenLength) {
            maxTokenLength = tokenList[i].length;
        }
    }
    //console.log(tokenList);
    //console.log(maxTokenLength);
    //console.log('vocab size', tokenizer.vocabSize)

    for (var i = 0; i < tokenList.length; i++) {
        for (var j = 1; j < tokenList[i].length; j++) {
            const tempSequence = tokenList[i].slice(0, j + 1);
            inputSequences.push(pad_array_left(tempSequence, maxTokenLength, 0));
        }
    }

    var X = [],
        Y = [];
    for (var i = 0; i < inputSequences.length; i++) {
        Y.push(inputSequences[i].pop());
        X.push(inputSequences[i]);
    }
    //one hot encoding the output
    Y = Y.map(n => oneHot(tokenizer.vocabSize, n));


    var { inputs, labels } = convertToTensor(X, Y)
    const model = createModel();
    await trainModel(model, inputs, labels);

    //saving the trained model in a file
    await model.save('file://./model-1a');


    //using the saved trained model to predict
    const loadedModel = await tf.loadLayersModel('file://./model-1a');
    console.log('Prediction from loaded model:');
    var outputTokenList = tokenizer.textsToSequences(['interactions in the PWA']);
    outputTokenList = [pad_array_left(outputTokenList[0], maxTokenLength - 1, 0)]
    var prediction = loadedModel.predict(tf.tensor(outputTokenList));
    prediction = prediction.reshape([tokenizer.vocabSize]);
    var { values, indices } = tf.topk(prediction, 3);
    indices = await indices.array();
    console.log(indices);
    console.log(tokenizer.indexWord[indices[0]], tokenizer.indexWord[indices[1]], tokenizer.indexWord[indices[2]])
}



async function trainModel(model, inputs, labels) {
    // Prepare the model for training.  
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const epochs = 250;
    const batchSize = 32;

    return await model.fit(inputs, labels, {
        epochs,
        batchSize,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log(epoch, logs);
            }
        }
    });
}
run()