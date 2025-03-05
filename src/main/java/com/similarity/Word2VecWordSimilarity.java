package com.similarity;

import javafx.util.Pair;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.common.Term;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.stream.Collectors;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadFactory;


/**
 * Word2Vec 词语相似度计算工具类
 * zmlm88@126.com
 */
public class Word2VecWordSimilarity {
    private static final Logger log = LoggerFactory.getLogger(Word2VecWordSimilarity.class);
    private static Word2Vec vec;
    private static final String _PATH = "D:\\program\\china.vec"; // 修改为你的实际路径

    // 添加静态Map来存储词向量
    static Map<String, INDArray> wordVectorsMap = new HashMap<>();

    // 保持现有的词向量相关字段
    private static final Map<String, float[]> wordVectors = new HashMap<>();
    private static final int VECTOR_SIZE = 300;

    // 添加词林相关字段
    private static final Map<String, Set<String>> wordCodesMap = new HashMap<>();
    private static final Map<String, Set<String>> codeWordsMap = new HashMap<>();

    static {
        try {
            initializeModel();
            // 添加词林加载
            initCilin();
        } catch (Exception e) {
            log.error("模型初始化失败", e);
            throw new RuntimeException("模型初始化失败", e);
        }
    }

    /**
     * 初始化模型
     */
    private static void initializeModel() throws IOException {
        log.info("开始加载模型...");
        File vectorFile = new File(_PATH);
        if (!vectorFile.exists()) {
            throw new FileNotFoundException("向量文件不存在: " + _PATH);
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(vectorFile), 8 * 1024 * 1024)) {
            log.info("成功打开文件，开始读取...");

            // 读取第一行获取维度信息
            String firstLine = reader.readLine();
            String[] dims = firstLine.split(" ");
            int vocabSize = Integer.parseInt(dims[0]);
            int vectorSize = Integer.parseInt(dims[1]);
            log.info("词汇量: {}, 向量维度: {}", vocabSize, vectorSize);

            // 预分配容量
            wordVectorsMap = new ConcurrentHashMap<>(vocabSize);

            // 读取所有行到内存
            log.info("开始读取词向量...");
            String line;
            int lineCount = 0;
            int successCount = 0;

            while ((line = reader.readLine()) != null) {
                lineCount++;
                try {
                    String[] tokens = line.trim().split("\\s+");
                    if (tokens.length != vectorSize + 1) {
                        continue;
                    }

                    float[] vector = new float[vectorSize];
                    for (int j = 0; j < vectorSize; j++) {
                        vector[j] = Float.parseFloat(tokens[j + 1]);
                    }
                    wordVectorsMap.put(tokens[0], Nd4j.create(vector));
                    successCount++;

                    if (lineCount % 10000 == 0) {
                        log.info("已处理 {} 行，成功加载 {} 个词向量", lineCount, successCount);
                    }
                } catch (Exception e) {
                    log.warn("处理第 {} 行时发生错误: {}", lineCount, e.getMessage());
                }
            }

            log.info("模型加载完成，总行数: {}，成功加载词向量: {}", lineCount, successCount);

            // 创建Word2Vec实例
            vec = new Word2Vec.Builder()
                    .layerSize(vectorSize)
                    .minWordFrequency(1)
                    .iterations(1)
                    .epochs(1)
                    .learningRate(0.025)
                    .windowSize(5)
                    .build();

            // 设置词汇表
            VocabCache<VocabWord> vocabCache = new AbstractCache<>();
            for (String word : wordVectorsMap.keySet()) {
                VocabWord vocabWord = new VocabWord(1.0, word);
                vocabCache.addToken(vocabWord);
                vocabCache.addWordToIndex(vocabCache.numWords(), word);
            }
            vec.setVocab(vocabCache);

        } catch (Exception e) {
            log.error("加载模型失败", e);
            throw new RuntimeException("加载模型失败", e);
        }
    }

    // 添加词林初始化方法
    private static void initCilin() throws IOException {
        String filePath = "data/cilin_expanded.txt";
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(
                        Word2VecWordSimilarity.class.getResourceAsStream(filePath), "UTF-8"))) {

            String line;
            int count = 0;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;

                String[] parts = line.split(" ");
                if (parts.length < 2) continue;

                String code = parts[0];
                Set<String> words = new HashSet<>(Arrays.asList(parts).subList(1, parts.length));

                codeWordsMap.put(code, words);
                for (String word : words) {
                    wordCodesMap.computeIfAbsent(word, k -> new HashSet<>()).add(code);
                }
                count++;
            }
            log.info("词林加载完成，共加载 {} 个编码", count);
        }
    }

    /**
     * 计算两个词的相似度
     */
    public static double calculateSimilarity(String text1, String text2) {
        try {
            log.info("开始计算文本相似度: '{}' vs '{}'", text1, text2);

            // 对输入文本进行分词，并过滤掉标点符号和停用词
            List<Term> terms1 = HanLP.segment(text1).stream()
                    .filter(term -> !term.nature.startsWith("w")  // 过滤标点符号
                            && !isStopWord(term.word))           // 过滤停用词
                    .collect(Collectors.toList());

            List<Term> terms2 = HanLP.segment(text2).stream()
                    .filter(term -> !term.nature.startsWith("w")
                            && !isStopWord(term.word))
                    .collect(Collectors.toList());

            log.info("分词结果(过滤后): text1={}, text2={}", terms1, terms2);

            // 获取所有词的向量并计算加权平均值
            INDArray vector1 = getWeightedAverageVector(terms1);
            INDArray vector2 = getWeightedAverageVector(terms2);

            if (vector1 == null || vector2 == null) {
                log.warn("无法为文本生成有效的向量表示: text1={}, text2={}",
                        vector1 == null ? "null" : "valid",
                        vector2 == null ? "null" : "valid");
                return 0.0;
            }

            // 计算余弦相似度
            double similarity = calculateCosineSimilarity(vector1, vector2);

            // 应用长度惩罚因子
            double lengthPenalty = calculateLengthPenalty(terms1.size(), terms2.size());
            similarity *= lengthPenalty;

            log.info("最终相似度计算完成: rawSimilarity={}, lengthPenalty={}, finalSimilarity={}",
                    similarity/lengthPenalty, lengthPenalty, similarity);
            return similarity;
        } catch (Exception e) {
            log.error("计算文本相似度失败: {} vs {}", text1, text2, e);
            return 0.0;
        }
    }

    private static boolean isStopWord(String word) {
        // 扩展停用词表
        Set<String> stopWords = new HashSet<>(Arrays.asList(
                "的", "了", "和", "是", "就", "都", "而", "及", "与", "着",
                "之", "用", "其", "中", "你", "我", "他", "她", "它", "要",
                "把", "被", "让", "在", "有", "个", "好", "这", "那", "什么",
                "啊", "哦", "呢", "吧", "啦", "么", "呀", "嘛", "哪", "那么",
                "这么", "怎么", "为", "以", "到", "得", "着", "过", "很", "对",
                "真", "的话", "所以", "因为", "但是", "不过", "可以", "现在"
        ));
        return stopWords.contains(word);
    }

    private static INDArray getWeightedAverageVector(List<Term> terms) {
        try {
            log.debug("开始计算加权平均向量，输入词项数: {}", terms.size());
            List<Pair<INDArray, Double>> vectorsWithWeights = new ArrayList<>();
            double totalWeight = 0.0;

            for (Term term : terms) {
                String word = term.word;
                INDArray vector = wordVectorsMap.get(word);
                if (vector != null) {
                    // 根据词性赋予不同权重
                    double weight = getTermWeight(term);
                    vectorsWithWeights.add(new Pair<>(vector, weight));
                    totalWeight += weight;
                    log.debug("找到词 '{}' 的向量，权重: {}", word, weight);
                } else {
                    log.debug("词 '{}' 不在词向量表中", word);
                }
            }

            if (vectorsWithWeights.isEmpty()) {
                log.warn("没有找到任何有效的词向量，输入词项: {}", terms);
                return null;
            }

            // 计算加权平均
            INDArray weightedSum = Nd4j.zeros(vectorsWithWeights.get(0).getKey().shape());
            for (Pair<INDArray, Double> pair : vectorsWithWeights) {
                weightedSum.addi(pair.getKey().mul(pair.getValue()));
            }
            return weightedSum.divi(totalWeight);
        } catch (Exception e) {
            log.error("计算加权平均向量失败，输入词项: {}", terms, e);
            return null;
        }
    }

    private static double getTermWeight(Term term) {
        // 根据词性赋予权重
        switch (term.nature.toString()) {
            case "n":   // 名词
            case "v":   // 动词
            case "a":   // 形容词
                return 1.0;
            case "t":   // 时间词
            case "s":   // 处所词
                return 0.8;
            case "f":   // 方位词
            case "b":   // 区别词
                return 0.6;
            default:
                return 0.4;
        }
    }

    private static double calculateCosineSimilarity(INDArray vector1, INDArray vector2) {
        double dotProduct = vector1.mul(vector2).sumNumber().doubleValue();
        double norm1 = vector1.norm2Number().doubleValue();
        double norm2 = vector2.norm2Number().doubleValue();
        return dotProduct / (norm1 * norm2);
    }

    private static double calculateLengthPenalty(int len1, int len2) {
        // 减轻长度差异惩罚
        double ratio = Math.min(len1, len2) / (double) Math.max(len1, len2);
        return Math.pow(ratio, 0.5);  // 使用平方根来减轻惩罚程度
    }

    /**
     * 判断两个词是否相似
     */
    public static boolean areWordsSimilar(String word1, String word2, double threshold) {
        try {
            log.info("开始判断词语相似性: word1='{}', word2='{}', threshold={}", word1, word2, threshold);

            if (word1 == null || word2 == null) {
                log.warn("输入词语为null: word1={}, word2={}", word1, word2);
                return false;
            }

            if (word1.equals(word2)) {
                log.info("词语完全相同，直接返回true");
                return true;
            }

            // 预处理：去除标点符号
            word1 = word1.replaceAll("[\\p{P}\\p{S}]", "");
            word2 = word2.replaceAll("[\\p{P}\\p{S}]", "");

            // 获取词性
            List<Term> terms1 = HanLP.segment(word1);
            List<Term> terms2 = HanLP.segment(word2);

            log.info("分词结果: terms1={}, terms2={}", terms1, terms2);

            // 获取主要词性
            String nature1 = terms1.isEmpty() ? "" : terms1.get(0).nature.toString();
            String nature2 = terms2.isEmpty() ? "" : terms2.get(0).nature.toString();

            // 计算相似度
            double similarity = calculateSimilarity(word1, word2);

            // 动态调整阈值
            double adjustedThreshold = threshold;

            // 如果都是名词，降低阈值
            if (nature1.startsWith("n") && nature2.startsWith("n")) {
                adjustedThreshold *= 0.8;
                log.info("检测到都是名词，降低阈值为: {}", adjustedThreshold);
            }

            // 如果都是动词，降低阈值
            if (nature1.startsWith("v") && nature2.startsWith("v")) {
                adjustedThreshold *= 0.8;
                log.info("检测到都是动词，降低阈值为: {}", adjustedThreshold);
            }

            // 如果词性不同，提高阈值
            if (!nature1.equals(nature2)) {
                adjustedThreshold *= 1.2;
                log.info("检测到词性不同，提高阈值为: {}", adjustedThreshold);
            }

            // 检查是否在同义词组中
            boolean inSynonymGroup = checkSynonymGroup(word1, word2);
            if (inSynonymGroup) {
                adjustedThreshold *= 0.7;
                log.info("检测到同义词关系，降低阈值为: {}", adjustedThreshold);
            }

            boolean isSimilar = similarity >= adjustedThreshold;

            log.info("相似度判断完成: rawSimilarity={}, adjustedThreshold={}, " +
                            "nature1={}, nature2={}, inSynonymGroup={}, isSimilar={}",
                    similarity, adjustedThreshold, nature1, nature2,
                    inSynonymGroup, isSimilar);

            return isSimilar;
        } catch (Exception e) {
            log.error("判断词语相似性失败: {} vs {}", word1, word2, e);
            return false;
        }
    }

    private static boolean checkSynonymGroup(String word1, String word2) {
        // 定义同义词组
        Set<Set<String>> synonymGroups = new HashSet<>();

        // 食物相关
        synonymGroups.add(new HashSet<>(Arrays.asList("面粉", "馒头", "面包", "饼")));
        synonymGroups.add(new HashSet<>(Arrays.asList("米", "大米", "米饭", "稻米")));
        synonymGroups.add(new HashSet<>(Arrays.asList("肉", "猪肉", "牛肉", "羊肉")));

        // 动作相关
        synonymGroups.add(new HashSet<>(Arrays.asList("走", "跑", "奔跑", "行走")));
        synonymGroups.add(new HashSet<>(Arrays.asList("说", "讲", "谈", "述说")));

        // 状态相关
        synonymGroups.add(new HashSet<>(Arrays.asList("快", "迅速", "急速", "飞快")));
        synonymGroups.add(new HashSet<>(Arrays.asList("慢", "缓慢", "迟缓", "慢速")));

        // 检查是否在同一个同义词组中
        return synonymGroups.stream()
                .anyMatch(group -> group.contains(word1) && group.contains(word2));
    }

    /**
     * 获取与给定词最相似的N个词
     */
    public static Collection<String> findSimilarWords(String text, int n) {
        try {
            log.info("开始查找与文本 '{}' 相似的 {} 个词", text, n);

            List<Term> terms = HanLP.segment(text);
            log.info("分词结果: {}", terms);

            INDArray queryVector = getWeightedAverageVector(terms);
            if (queryVector == null) {
                log.warn("无法为文本 '{}' 生成有效的向量表示", text);
                return Collections.emptyList();
            }
            log.info("成功生成查询向量");

            // 使用并行流和ConcurrentHashMap来加速计算
            log.info("开始并行计算相似度...");
            ConcurrentHashMap<String, Double> similarities = new ConcurrentHashMap<>(wordVectorsMap.size());
            AtomicInteger processedCount = new AtomicInteger(0);

            // 将词向量分批处理
            int batchSize = 1000;
            List<List<Map.Entry<String, INDArray>>> batches = new ArrayList<>();
            List<Map.Entry<String, INDArray>> currentBatch = new ArrayList<>();

            for (Map.Entry<String, INDArray> entry : wordVectorsMap.entrySet()) {
                currentBatch.add(entry);
                if (currentBatch.size() == batchSize) {
                    batches.add(new ArrayList<>(currentBatch));
                    currentBatch.clear();
                }
            }
            if (!currentBatch.isEmpty()) {
                batches.add(currentBatch);
            }

            // 并行处理每个批次
            batches.parallelStream().forEach(batch -> {
                for (Map.Entry<String, INDArray> entry : batch) {
                    String word = entry.getKey();
                    INDArray vector = entry.getValue();

                    // 批量计算相似度
                    double similarity = queryVector.mul(vector).sumNumber().doubleValue() /
                            (queryVector.norm2Number().doubleValue() * vector.norm2Number().doubleValue());
                    similarities.put(word, similarity);

                    int count = processedCount.incrementAndGet();
                    if (count % 10000 == 0) {
                        log.info("已处理 {} / {} 个词向量", count, wordVectorsMap.size());
                    }
                }
            });

            log.info("相似度计算完成，开始排序...");

            // 使用优先队列来获取top N，避免全量排序
            PriorityQueue<Map.Entry<String, Double>> topN = new PriorityQueue<>(
                    n + 1, Map.Entry.<String, Double>comparingByValue()
            );

            for (Map.Entry<String, Double> entry : similarities.entrySet()) {
                topN.offer(entry);
                if (topN.size() > n) {
                    topN.poll();
                }
            }

            // 转换结果
            List<String> result = new ArrayList<>(n);
            while (!topN.isEmpty()) {
                result.add(0, topN.poll().getKey());
            }

            log.info("找到 {} 个相似词: {}", result.size(), result);
            return result;
        } catch (Exception e) {
            log.error("查找相似词失败: {}", text, e);
            return Collections.emptyList();
        }
    }

    /**
     * 检查词是否在词汇表中
     */
    public static boolean hasWord(String word) {
        return vec.hasWord(word);
    }

    /**
     * 获取词向量
     */
    public static double[] getWordVector(String word) {
        try {
            return vec.getWordVector(word);
        } catch (Exception e) {
            log.error("获取词向量失败: {}", word, e);
            return null;
        }
    }

    private static boolean isQuestion(String text) {
        // 问号判断
        if (text.contains("?") || text.contains("？")) {
            return true;
        }

        // 疑问词判断
        Set<String> questionWords = new HashSet<>(Arrays.asList(
                "什么", "怎么", "怎样", "如何", "哪", "谁", "为什么", "几",
                "多少", "是否", "能否", "可否", "吗", "呢", "吧", "啊",
                "嘛", "呀", "哪里", "哪儿", "何时", "为何", "多久"
        ));

        // 分词后检查是否包含疑问词
        List<Term> terms = HanLP.segment(text);
        for (Term term : terms) {
            if (questionWords.contains(term.word)) {
                log.debug("检测到疑问词: {}", term.word);
                return true;
            }
        }

        // 语气词判断（句尾）
        if (terms.size() > 0) {
            String lastWord = terms.get(terms.size() - 1).word;
            Set<String> questionTones = new HashSet<>(Arrays.asList(
                    "吗", "呢", "吧", "啊", "嘛", "呀", "么"
            ));
            if (questionTones.contains(lastWord)) {
                log.debug("检测到句尾疑问语气词: {}", lastWord);
                return true;
            }
        }

        // 特殊句式判断
        String[] questionPatterns = {
                "是不是", "对不对", "行不行", "要不要", "能不能", "可不可以",
                "有没有", "对吧", "是吧", "好吧"
        };
        for (String pattern : questionPatterns) {
            if (text.contains(pattern)) {
                log.debug("检测到疑问句式: {}", pattern);
                return true;
            }
        }

        log.debug("未检测到问句特征");
        return false;
    }

    // 添加词林相似度计算方法
    private static double calculateCilinSimilarity(String word1, String word2) {
        if (word1.equals(word2)) return 1.0;

        Set<String> codes1 = wordCodesMap.getOrDefault(word1, Collections.emptySet());
        Set<String> codes2 = wordCodesMap.getOrDefault(word2, Collections.emptySet());

        if (codes1.isEmpty() || codes2.isEmpty()) {
            return -1.0;  // 表示无法通过词林计算相似度
        }

        double maxSimilarity = 0.0;
        for (String code1 : codes1) {
            for (String code2 : codes2) {
                double similarity = calculateCodeSimilarity(code1, code2);
                maxSimilarity = Math.max(maxSimilarity, similarity);
            }
        }

        return maxSimilarity;
    }

    private static double calculateCodeSimilarity(String code1, String code2) {
        if (code1.equals(code2)) return 1.0;

        int commonLength = 0;
        int minLength = Math.min(code1.length(), code2.length());

        for (int i = 0; i < minLength; i++) {
            if (code1.charAt(i) == code2.charAt(i)) {
                commonLength++;
            } else {
                break;
            }
        }

        // 根据词林的层级结构计算相似度
        switch (commonLength) {
            case 0: return 0.0;   // 不同大类
            case 1: return 0.1;   // 同属一个大类
            case 2: return 0.2;   // 同属一个中类
            case 3: return 0.4;   // 同属一个小类
            case 4: return 0.6;   // 同属一个词群
            case 5: return 0.8;   // 同属一个原子词群
            default: return 0.9;  // 更深层次的相似
        }
    }

    // 将原有的相似度计算方法重命名为calculateVectorSimilarity
    private static double calculateVectorSimilarity(String word1, String word2) {
        // ... 原有的词向量相似度计算逻辑 ...
        // 保持原有实现不变
        return 0.0; // 这里应该是原有的实现
    }

    // 使用示例
    public static void main(String[] args) {
        try {
            // 计算相似度
            String word1 = "面粉";
            String word2 = "馒头";
            //  double similarity = calculateSimilarity(word1, word2);
            //System.out.println("相似度: " + similarity);

            // 查找相似词
//            Collection<String> similarWords = findSimilarWords(word1, 5);
//            System.out.println("相似词: " + similarWords);

            // 判断是否相似
            boolean isSimilar = areWordsSimilar(word1, word2, 0.7);
            System.out.println("是否相似: " + isSimilar);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}