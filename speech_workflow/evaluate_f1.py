import re
from typing import Dict, List, Tuple, Set
from collections import Counter
from difflib import SequenceMatcher


TAG_PATTERN = r'\[(IM|RM|R|D)\](.*?)\[\1\]'
TOKEN_PATTERN = r"\w+(?:['-]\w+)?|[^\w\s]"


def normalize_text(text: str) -> str:
    """文本归一化处理"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    return text


def tokenize_text(text: str) -> List[str]:
    """按词/标点切分，用于重复内容的逐实例对齐。"""
    return re.findall(TOKEN_PATTERN, normalize_text(text))


def is_word_token(token: str) -> bool:
    """是否按单词计入 token 级指标；标点不计入。"""
    return bool(re.search(r"\w", token))


def metric_tokens(tokens: List[str]) -> List[str]:
    """指标用 token：只保留单词，大小写归一。"""
    return [token.lower() for token in tokens if is_word_token(token)]


def strip_inner_annotation_tags(text: str) -> str:
    """删除嵌套在标签内容里的标注符号，只保留真实文本。"""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(TAG_PATTERN, r'\2', text)
    return normalize_text(text)


def attach_token_indices(annotated_text: str, annotations: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    给每个标注补充它在去标注参考文本中的 token 位置。

    这样像多个 "like" 这类重复内容不会被全局子串搜索混在一起。
    """
    reference_tokens = []
    target_tokens = []
    last_end = 0
    matches = list(re.finditer(TAG_PATTERN, annotated_text))

    for idx, match in enumerate(matches):
        normal_tokens = tokenize_text(annotated_text[last_end:match.start()])
        reference_tokens.extend(normal_tokens)
        target_tokens.extend(normal_tokens)

        tag_content = annotations[idx].get('content', match.group(2).strip()) if idx < len(annotations) else match.group(2).strip()
        tag_tokens = tokenize_text(tag_content)
        start = len(reference_tokens)
        reference_tokens.extend(tag_tokens)
        end = len(reference_tokens)

        if idx < len(annotations):
            annotations[idx]['tokens'] = tag_tokens
            annotations[idx]['token_indices'] = list(range(start, end))
            if annotations[idx]['type'] == 'R':
                target_start = len(target_tokens)
                target_tokens.extend(tag_tokens)
                annotations[idx]['target_token_indices'] = list(range(target_start, len(target_tokens)))
            else:
                annotations[idx]['target_insert_pos'] = len(target_tokens)

        last_end = match.end()

    tail_tokens = tokenize_text(annotated_text[last_end:])
    reference_tokens.extend(tail_tokens)
    target_tokens.extend(tail_tokens)
    return reference_tokens, target_tokens


def align_reference_tokens(reference_tokens: List[str], processed_text: str) -> Set[int]:
    """把参考文本 token 和模型输出 token 做序列对齐，返回仍保留的参考 token 位置。"""
    processed_tokens = tokenize_text(processed_text)
    matcher = SequenceMatcher(None, reference_tokens, processed_tokens, autojunk=False)
    matched_indices = set()

    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == 'equal':
            matched_indices.update(range(i1, i2))

    return matched_indices


def tag_match_stats(tag: Dict, matched_indices: Set[int]) -> Dict:
    """返回某个标注内容在模型输出中被对齐保留下来的 token 统计。"""
    token_indices = tag.get('token_indices', [])
    tokens = tag.get('tokens', tokenize_text(tag.get('content', '')))
    matched_count = sum(1 for idx in token_indices if idx in matched_indices)

    return {
        'matched_count': matched_count,
        'total_tokens': len(token_indices),
        'any_present': matched_count > 0,
        'fully_present': bool(token_indices) and matched_count == len(token_indices),
        'tokens': tokens
    }


def count_token_sequence(tokens: List[str], sequence: List[str]) -> int:
    """统计 token 序列连续出现的次数。"""
    if not sequence or len(sequence) > len(tokens):
        return 0

    seq_len = len(sequence)
    return sum(1 for i in range(len(tokens) - seq_len + 1) if tokens[i:i + seq_len] == sequence)


def tag_count_stats(tag: Dict, processed_tokens: List[str], target_tokens: List[str]) -> Dict:
    """按 Ground Truth 次数判断某个标注内容是否多留/少留。"""
    tokens = tag.get('tokens', tokenize_text(tag.get('content', '')))
    processed_count = count_token_sequence(processed_tokens, tokens)
    target_count = count_token_sequence(target_tokens, tokens)

    return {
        'processed_count': processed_count,
        'target_count': target_count,
        'extra_count': max(0, processed_count - target_count),
        'missing_count': max(0, target_count - processed_count),
        'tokens': tokens
    }


def tag_token_extra_stats(tag: Dict, processed_tokens: List[str], target_tokens: List[str]) -> Dict:
    """按 token 出现次数判断删除类标签是否有任何内容残留，主要用于 D 长段落。"""
    tokens = tag.get('tokens', tokenize_text(tag.get('content', '')))
    tag_counter = Counter(tokens)
    processed_counter = Counter(processed_tokens)
    target_counter = Counter(target_tokens)

    extra_tokens = {}
    for token in tag_counter:
        extra = max(0, processed_counter[token] - target_counter[token])
        if extra > 0:
            extra_tokens[token] = extra

    return {
        'extra_token_count': sum(extra_tokens.values()),
        'extra_tokens': extra_tokens,
        'tokens': tokens
    }


def parse_annotated_text(annotated_text: str) -> Tuple[str, List[Dict], Set[str]]:
    """
    解析标注文本，提取标注信息
    
    Args:
        annotated_text: 带标注的文本
        
    Returns:
        (pure_text, annotations, should_keep_contents): 
        纯文本, 标注列表, 应该保留的内容集合
    """
    pattern = TAG_PATTERN
    annotations = []
    
    for match in re.finditer(pattern, annotated_text):
        tag_type = match.group(1)
        content = strip_inner_annotation_tags(match.group(2).strip()) if tag_type == 'D' else match.group(2).strip()
        annotations.append({
            'type': tag_type,
            'content': content,
            'start': match.start(),
            'end': match.end()
        })
    
    pure_text = re.sub(pattern, r'\2', annotated_text)
    pure_text = normalize_text(pure_text)
    
    should_keep = set()

    for anno in annotations:
        if anno['type'] == 'R' and anno['content'].strip():
            should_keep.add(anno['content'].strip())
    
    temp_text = annotated_text
    temp_text = re.sub(r'\[(IM|RM|D)\](.*?)\[\1\]', '', temp_text)
    temp_text = re.sub(r'\[R\](.*?)\[R\]', r'\1', temp_text)
    temp_text = normalize_text(temp_text)
    
    words = re.split(r'[\s,.!?]+', temp_text)
    for word in words:
        if word.strip():
            should_keep.add(word.strip())
    
    return pure_text, annotations, should_keep


def generate_ground_truth(annotated_text: str) -> str:
    """从标注文本生成 Ground Truth"""
    result = annotated_text
    pattern = r'\[(IM|RM|D)\](.*?)\[\1\]'
    result = re.sub(pattern, '', result)
    pattern = r'\[R\](.*?)\[R\]'
    result = re.sub(pattern, r'\1', result)
    result = normalize_text(result)
    return result


def extract_annotations_with_positions(annotated_text: str) -> List[Dict]:
    """提取标注及其位置，用于找到应该删除的内容"""
    pattern = TAG_PATTERN
    annotations = []
    
    offset = 0
    modified_text = annotated_text
    
    while True:
        match = re.search(pattern, modified_text)
        if not match:
            break
        
        tag_type = match.group(1)
        content = match.group(2).strip()
        start = match.start() + offset
        end = match.end() + offset
        
        annotations.append({
            'type': tag_type,
            'content': content,
            'start': start,
            'end': end,
            'full_match': match.group(0)
        })
        
        offset += match.end()
        modified_text = modified_text[match.end():]
    
    return annotations


def evaluate_delete_tags(
    processed_text: str,
    annotations: List[Dict],
    tag_type: str,
    reference_tokens: List[str] = None,
    target_tokens: List[str] = None,
    token_level_delete: bool = False
) -> Dict:
    """评估需要删除的单标签，如 IM/D。"""
    processed_norm = normalize_text(processed_text)
    matched_indices = align_reference_tokens(reference_tokens, processed_text) if reference_tokens else None
    processed_tokens = tokenize_text(processed_text)
    
    delete_tags = [a for a in annotations if a['type'] == tag_type]
    tp = 0
    fn = 0
    details = []
    
    for tag in delete_tags:
        content = tag['content'].strip()
        
        if not content:
            tp += 1
            details.append({'content': content, 'in_result': False, 'success': True})
            continue
        
        if target_tokens is not None and tag.get('tokens') is not None:
            if token_level_delete:
                stats = tag_token_extra_stats(tag, processed_tokens, target_tokens)
                in_result = stats['extra_token_count'] > 0
            else:
                stats = tag_count_stats(tag, processed_tokens, target_tokens)
                in_result = stats['extra_count'] > 0
            matched_stats = None
        elif matched_indices is not None and tag.get('token_indices') is not None:
            matched_stats = tag_match_stats(tag, matched_indices)
            in_result = matched_stats['any_present']
            stats = None
        else:
            stats = None
            matched_stats = None
            in_result = content in processed_norm
        
        if not in_result:
            tp += 1
            success = True
        else:
            fn += 1
            success = False
        
        details.append({
            'content': content,
            'in_result': in_result,
            'success': success,
            'matched_tokens': matched_stats['matched_count'] if matched_stats else None,
            'total_tokens': matched_stats['total_tokens'] if matched_stats else None,
            'processed_count': stats.get('processed_count') if stats else None,
            'target_count': stats.get('target_count') if stats else None,
            'extra_token_count': stats.get('extra_token_count') if stats else None,
            'extra_tokens': stats.get('extra_tokens') if stats else None
        })
    
    total = tp + fn
    recall = tp / total if total > 0 else 1.0
    
    return {
        'tp': tp,
        'fn': fn,
        'total': total,
        'recall': recall,
        'details': details
    }


def evaluate_im_tags(
    processed_text: str,
    annotations: List[Dict],
    reference_tokens: List[str] = None,
    target_tokens: List[str] = None
) -> Dict:
    """评估 IM 标签"""
    return evaluate_delete_tags(processed_text, annotations, 'IM', reference_tokens, target_tokens)


def evaluate_d_tags(
    processed_text: str,
    annotations: List[Dict],
    reference_tokens: List[str] = None,
    target_tokens: List[str] = None
) -> Dict:
    """评估 D 标签：主观应删内容，单独统计，不并入 IM/RM→R 总体指标。"""
    return evaluate_delete_tags(processed_text, annotations, 'D', reference_tokens, target_tokens, token_level_delete=True)


def evaluate_rm_r_pairs(
    processed_text: str,
    annotations: List[Dict],
    reference_tokens: List[str] = None,
    target_tokens: List[str] = None
) -> Dict:
    """评估 RM→R 对。"""
    processed_norm = normalize_text(processed_text)
    matched_indices = align_reference_tokens(reference_tokens, processed_text) if reference_tokens else None
    processed_tokens = tokenize_text(processed_text)
    
    rm_r_pairs = []
    i = 0
    while i < len(annotations):
        if i + 1 < len(annotations):
            curr = annotations[i]
            next_anno = annotations[i + 1]
            
            if curr['type'] == 'RM' and next_anno['type'] == 'R':
                rm_r_pairs.append({
                    'rm': curr,
                    'r': next_anno
                })
                i += 2
                continue
        
        i += 1
    
    tp = 0
    fn = 0
    details = []
    
    for pair in rm_r_pairs:
        rm_content = pair['rm']['content'].strip()
        r_content = pair['r']['content'].strip()
        
        if target_tokens is not None:
            rm_stats = tag_count_stats(pair['rm'], processed_tokens, target_tokens)
            r_stats = tag_count_stats(pair['r'], processed_tokens, target_tokens)
            rm_in_result = rm_stats['extra_count'] > 0
            r_in_result = r_stats['missing_count'] == 0
            matched_rm_stats = None
            matched_r_stats = None
        elif matched_indices is not None:
            matched_rm_stats = tag_match_stats(pair['rm'], matched_indices)
            matched_r_stats = tag_match_stats(pair['r'], matched_indices)
            rm_in_result = matched_rm_stats['any_present']
            r_in_result = matched_r_stats['fully_present']
            rm_stats = None
            r_stats = None
        else:
            rm_stats = None
            r_stats = None
            matched_rm_stats = None
            matched_r_stats = None
            rm_in_result = bool(rm_content and rm_content in processed_norm)
            r_in_result = bool(r_content and r_content in processed_norm)
        
        success = False
        if rm_content and r_content:
            if target_tokens is not None:
                success = (not rm_in_result) and r_in_result
            elif matched_indices is not None and rm_content == r_content:
                # Identical RM/R text is correct only when one occurrence remains.
                expected_kept = matched_r_stats['total_tokens'] or matched_rm_stats['total_tokens']
                actual_kept = matched_rm_stats['matched_count'] + matched_r_stats['matched_count']
                success = actual_kept == expected_kept
            elif matched_indices is not None:
                success = (not rm_in_result) and r_in_result
            elif rm_content == r_content:
                success = r_in_result or rm_in_result
            else:
                if not rm_in_result or not r_in_result:
                    success = True
        elif rm_content and not r_content:
            success = not rm_in_result
        elif r_content and not rm_content:
            success = r_in_result
        else:
            success = True
        
        if success:
            tp += 1
        else:
            fn += 1
        
        details.append({
            'rm_content': rm_content,
            'r_content': r_content,
            'rm_in_result': rm_in_result,
            'r_in_result': r_in_result,
            'success': success,
            'rm_matched_tokens': matched_rm_stats['matched_count'] if matched_rm_stats else None,
            'rm_total_tokens': matched_rm_stats['total_tokens'] if matched_rm_stats else None,
            'r_matched_tokens': matched_r_stats['matched_count'] if matched_r_stats else None,
            'r_total_tokens': matched_r_stats['total_tokens'] if matched_r_stats else None,
            'rm_processed_count': rm_stats['processed_count'] if rm_stats else None,
            'rm_target_count': rm_stats['target_count'] if rm_stats else None,
            'r_processed_count': r_stats['processed_count'] if r_stats else None,
            'r_target_count': r_stats['target_count'] if r_stats else None
        })
    
    total = tp + fn
    recall = tp / total if total > 0 else 1.0
    
    return {
        'tp': tp,
        'fn': fn,
        'total': total,
        'recall': recall,
        'details': details
    }


def detect_false_negatives(processed_text: str, should_keep: Set[str]) -> Dict:
    """
    检测误删（False Negatives）：应该保留的内容被删掉了
    
    Args:
        processed_text: 模型处理结果
        should_keep: 应该保留的内容集合
        
    Returns:
        (fp_content_list, fp_count)
    """
    processed_norm = normalize_text(processed_text)
    
    fp_contents = []
    for content in should_keep:
        if content and content not in processed_norm:
            fp_contents.append(content)
    
    return {
        'fp_contents': fp_contents,
        'fp_count': len(fp_contents)
    }


def calculate_token_overdelete_fp(processed_text: str, target_tokens: List[str]) -> Dict:
    """按单词数统计多删：GT 应保留的 token 在模型输出中缺失的数量。"""
    processed_counter = Counter(metric_tokens(tokenize_text(processed_text)))
    target_counter = Counter(metric_tokens(target_tokens or []))

    fp_tokens = {}
    for token, expected_count in target_counter.items():
        missing_count = max(0, expected_count - processed_counter[token])
        if missing_count > 0:
            fp_tokens[token] = missing_count

    return {
        'fp': sum(fp_tokens.values()),
        'fp_tokens': fp_tokens
    }


def calculate_d_token_metrics(processed_text: str, annotations: List[Dict], target_tokens: List[str]) -> Dict:
    """
    D 单独按单词级计算：
    - TP：D 中应删单词被删掉的数量
    - FN：D 中应删单词仍额外残留的数量
    - FP：非 D 应保留单词被多删的数量
    """
    processed_counter = Counter(metric_tokens(tokenize_text(processed_text)))
    target_counter = Counter(metric_tokens(target_tokens or []))

    d_tokens = []
    for tag in annotations:
        if tag['type'] == 'D':
            d_tokens.extend(metric_tokens(tag.get('tokens', tokenize_text(tag.get('content', '')))))

    d_counter = Counter(d_tokens)
    tp = 0
    fn = 0
    residual_tokens = {}

    for token, expected_delete_count in d_counter.items():
        extra_present = max(0, processed_counter[token] - target_counter[token])
        missed_delete_count = min(expected_delete_count, extra_present)
        deleted_count = expected_delete_count - missed_delete_count
        tp += deleted_count
        fn += missed_delete_count
        if missed_delete_count > 0:
            residual_tokens[token] = missed_delete_count

    fp_result = calculate_token_overdelete_fp(processed_text, target_tokens)
    total = tp + fn
    recall = tp / total if total > 0 else 1.0
    precision = calculate_precision(tp, fp_result['fp'])
    f1 = calculate_f1(precision, recall)

    return {
        'tp': tp,
        'fn': fn,
        'fp': fp_result['fp'],
        'total': total,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'residual_tokens': residual_tokens,
        'fp_tokens': fp_result['fp_tokens']
    }


def calculate_precision(tp: int, fp: int) -> float:
    """计算 Precision"""
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)


def calculate_f1(precision: float, recall: float) -> float:
    """计算 F1"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(annotated_path: str, processed_path: str) -> Dict:
    """主评价函数"""
    with open(annotated_path, 'r', encoding='utf-8') as f:
        annotated_text = f.read()
    
    with open(processed_path, 'r', encoding='utf-8') as f:
        processed_text = f.read()
    
    _, annotations, should_keep = parse_annotated_text(annotated_text)
    reference_tokens, target_tokens = attach_token_indices(annotated_text, annotations)
    ground_truth = generate_ground_truth(annotated_text)
    
    im_result = evaluate_im_tags(processed_text, annotations, reference_tokens, target_tokens)

    d_result = evaluate_d_tags(processed_text, annotations, reference_tokens, target_tokens)
    d_token_metrics = calculate_d_token_metrics(processed_text, annotations, target_tokens)

    rm_r_result = evaluate_rm_r_pairs(processed_text, annotations, reference_tokens, target_tokens)

    fp_result = detect_false_negatives(processed_text, should_keep)
    
    total_fp = fp_result['fp_count']
    
    im_precision = calculate_precision(im_result['tp'], total_fp)
    im_f1 = calculate_f1(im_precision, im_result['recall'])
    
    rm_r_precision = calculate_precision(rm_r_result['tp'], total_fp)
    rm_r_f1 = calculate_f1(rm_r_precision, rm_r_result['recall'])
    
    processed_norm = normalize_text(processed_text)
    gt_norm = normalize_text(ground_truth)
    exact_match = 1.0 if processed_norm == gt_norm else 0.0
    
    return {
        'exact_match': exact_match,
        'im': {
            'tp': im_result['tp'],
            'fn': im_result['fn'],
            'total': im_result['total'],
            'recall': im_result['recall'],
            'precision': im_precision,
            'f1': im_f1,
            'details': im_result['details']
        },
        'rm_r': {
            'tp': rm_r_result['tp'],
            'fn': rm_r_result['fn'],
            'total': rm_r_result['total'],
            'recall': rm_r_result['recall'],
            'precision': rm_r_precision,
            'f1': rm_r_f1,
            'details': rm_r_result['details']
        },
        'd': {
            'tp': d_token_metrics['tp'],
            'fn': d_token_metrics['fn'],
            'fp': d_token_metrics['fp'],
            'total': d_token_metrics['total'],
            'recall': d_token_metrics['recall'],
            'precision': d_token_metrics['precision'],
            'f1': d_token_metrics['f1'],
            'details': d_result['details'],
            'tag_tp': d_result['tp'],
            'tag_fn': d_result['fn'],
            'tag_total': d_result['total'],
            'residual_tokens': d_token_metrics['residual_tokens'],
            'fp_tokens': d_token_metrics['fp_tokens']
        },
        'false_positives': {
            'count': fp_result['fp_count'],
            'contents': fp_result['fp_contents']
        },
        'ground_truth': ground_truth,
        'processed_text': processed_text
    }


def print_evaluation_report(results: Dict):
    """打印评价报告"""
    print("=" * 80)
    print("口播去水词评价报告")
    print("=" * 80)
    print()
    
    print(f"Exact Match: {'✓' if results['exact_match'] == 1 else '✗'} ({results['exact_match'] * 100:.2f}%)")
    print()
    
    print("-" * 80)
    print("【IM 水词指标】")
    print(f"  TP(正确删除): {results['im']['tp']}/{results['im']['total']}")
    print(f"  FN(漏删):     {results['im']['fn']}")
    print(f"  Recall:       {results['im']['recall'] * 100:.2f}%")
    print(f"  Precision:    {results['im']['precision'] * 100:.2f}%")
    print(f"  F1:           {results['im']['f1'] * 100:.2f}%")
    
    if results['im']['details']:
        print("  详情:")
        for detail in results['im']['details'][:5]:
            status = "✓" if detail['success'] else "✗"
            print(f"    {status} '{detail['content']}' - {'已删除' if detail['success'] else '仍在结果中'}")
        if len(results['im']['details']) > 5:
            print(f"    ... 还有 {len(results['im']['details']) - 5} 个")
    
    print()
    
    print("-" * 80)
    print("【RM→R 对指标】")
    print(f"  TP(正确处理): {results['rm_r']['tp']}/{results['rm_r']['total']}")
    print(f"  FN(漏处理):   {results['rm_r']['fn']}")
    print(f"  Recall:       {results['rm_r']['recall'] * 100:.2f}%")
    print(f"  Precision:    {results['rm_r']['precision'] * 100:.2f}%")
    print(f"  F1:           {results['rm_r']['f1'] * 100:.2f}%")
    
    if results['rm_r']['details']:
        print("  详情:")
        for detail in results['rm_r']['details']:
            status = "✓" if detail['success'] else "✗"
            print(f"    {status} RM='{detail['rm_content']}' R='{detail['r_content']}'")
            print(f"       - RM在结果: {detail['rm_in_result']}, R在结果: {detail['r_in_result']}")
    
    print()
    
    if 'd' in results:
        print("-" * 80)
        print("【D 主观应删指标】")
        print(f"  TP(正确删除词): {results['d']['tp']}/{results['d']['total']}")
        print(f"  FN(漏删词):     {results['d']['fn']}")
        print(f"  FP(多删词):     {results['d'].get('fp', 0)}")
        print(f"  Recall:       {results['d']['recall'] * 100:.2f}%")
        print(f"  Precision:    {results['d']['precision'] * 100:.2f}%")
        print(f"  F1:           {results['d']['f1'] * 100:.2f}%")
        if 'tag_total' in results['d']:
            print(f"  D段级统计:    {results['d']['tag_tp']}/{results['d']['tag_total']} 段删干净")

        if results['d']['details']:
            print("  详情:")
            for detail in results['d']['details'][:5]:
                status = "✓" if detail['success'] else "✗"
                print(f"    {status} '{detail['content']}' - {'已删除' if detail['success'] else '仍在结果中'}")
            if len(results['d']['details']) > 5:
                print(f"    ... 还有 {len(results['d']['details']) - 5} 个")
        print()

    print("-" * 80)
    print("【误删检测】")
    print(f"  FP(误删数): {results['false_positives']['count']}")
    if results['false_positives']['contents']:
        print("  误删内容:")
        for content in results['false_positives']['contents'][:10]:
            print(f"    - '{content}'")
        if len(results['false_positives']['contents']) > 10:
            print(f"    ... 还有 {len(results['false_positives']['contents']) - 10} 个")
    print()
    
    print("-" * 80)
    print("【Ground Truth】")
    print(results['ground_truth'][:200] + "..." if len(results['ground_truth']) > 200 else results['ground_truth'])
    print()
    
    print("-" * 80)
    print("【模型输出】")
    print(results['processed_text'][:200] + "..." if len(results['processed_text']) > 200 else results['processed_text'])
    print()
    print("=" * 80)


def parse_annotated_text_v2(annotated_text):
    """
    解析标注文本，提取标注信息（新版本，包含上下文）
    
    Args:
        annotated_text: 带标注的文本
        
    Returns:
        (pure_text, annotations, should_keep_contents): 
        纯文本, 标注列表（包含上下文）, 应该保留的内容集合
    """
    pattern = TAG_PATTERN
    
    pure_text_with_positions = []
    last_end = 0
    annotations = []
    
    matches = list(re.finditer(pattern, annotated_text))
    
    for i, match in enumerate(matches):
        tag_type = match.group(1)
        content = strip_inner_annotation_tags(match.group(2).strip()) if tag_type == 'D' else match.group(2).strip()
        
        if match.start() > last_end:
            normal_text = annotated_text[last_end:match.start()]
            pure_text_with_positions.append((normal_text, None, None))
        
        pure_text_with_positions.append((content, tag_type, i))

        last_end = match.end()
        
        annotations.append({
            'type': tag_type,
            'content': content,
            'start': match.start(),
            'end': match.end()
        })
    
    if last_end < len(annotated_text):
        normal_text = annotated_text[last_end:]
        pure_text_with_positions.append((normal_text, None, None))
    
    pure_text = ''
    tag_positions_in_pure = []
    
    for text_part, tag_type, tag_idx in pure_text_with_positions:
        if tag_type is None:
            pure_text += text_part
        else:
            tag_start = len(pure_text)
            pure_text += text_part
            tag_end = len(pure_text)
            tag_positions_in_pure.append({
                'tag_idx': tag_idx,
                'tag_type': tag_type,
                'content': text_part,
                'start_in_pure': tag_start,
                'end_in_pure': tag_end
            })
    
    pure_text = normalize_text(pure_text)

    for pos_info in tag_positions_in_pure:
        tag_idx = pos_info['tag_idx']
        
        start_ctx = max(0, pos_info['start_in_pure'] - 30)
        end_ctx = min(len(pure_text), pos_info['end_in_pure'] + 30)
        context_before = pure_text[start_ctx:pos_info['start_in_pure']]
        context_after = pure_text[pos_info['end_in_pure']:end_ctx]
        
        annotations[tag_idx]['context_before'] = context_before
        annotations[tag_idx]['context_after'] = context_after
        annotations[tag_idx]['full_context'] = context_before + pos_info['content'] + context_after
    
    should_keep = set()

    for anno in annotations:
        if anno['type'] == 'R' and anno['content'].strip():
            should_keep.add(anno['content'].strip())
    
    temp_text = annotated_text
    temp_text = re.sub(r'\[(IM|RM|D)\](.*?)\[\1\]', '', temp_text)
    temp_text = re.sub(r'\[R\](.*?)\[R\]', r'\1', temp_text)
    temp_text = normalize_text(temp_text)
    words = re.split(r'[\s,.!?]+', temp_text)
    for word in words:
        if word.strip():
            should_keep.add(word.strip())
    
    return pure_text, annotations, should_keep


def evaluate_im_tags_v2(processed_text, annotations, reference_tokens=None, target_tokens=None):
    """评估 IM 标签（基于 token 序列对齐，避免重复内容误判）"""
    return evaluate_im_tags(processed_text, annotations, reference_tokens, target_tokens)


def evaluate_d_tags_v2(processed_text, annotations, reference_tokens=None, target_tokens=None):
    """评估 D 标签（基于 token 序列对齐，单独统计主观应删内容）"""
    return evaluate_d_tags(processed_text, annotations, reference_tokens, target_tokens)


def evaluate_rm_r_pairs_v2(processed_text, annotations, reference_tokens=None, target_tokens=None):
    """评估 RM→R 对（基于 token 序列对齐，避免重复内容误判）。"""
    return evaluate_rm_r_pairs(processed_text, annotations, reference_tokens, target_tokens)


def evaluate_v2(annotated_path, processed_path):
    """主评价函数（新版本：基于上下文匹配）"""
    with open(annotated_path, 'r', encoding='utf-8') as f:
        annotated_text = f.read()
    
    with open(processed_path, 'r', encoding='utf-8') as f:
        processed_text = f.read()
    
    _, annotations, should_keep = parse_annotated_text_v2(annotated_text)
    reference_tokens, target_tokens = attach_token_indices(annotated_text, annotations)
    ground_truth = generate_ground_truth(annotated_text)
    
    im_result = evaluate_im_tags_v2(processed_text, annotations, reference_tokens, target_tokens)

    d_result = evaluate_d_tags_v2(processed_text, annotations, reference_tokens, target_tokens)
    d_token_metrics = calculate_d_token_metrics(processed_text, annotations, target_tokens)

    rm_r_result = evaluate_rm_r_pairs_v2(processed_text, annotations, reference_tokens, target_tokens)

    fp_result = detect_false_negatives(processed_text, should_keep)
    
    total_fp = fp_result['fp_count']
    
    im_precision = calculate_precision(im_result['tp'], total_fp)
    im_f1 = calculate_f1(im_precision, im_result['recall'])
    
    rm_r_precision = calculate_precision(rm_r_result['tp'], total_fp)
    rm_r_f1 = calculate_f1(rm_r_precision, rm_r_result['recall'])
    
    processed_norm = normalize_text(processed_text)
    gt_norm = normalize_text(ground_truth)
    exact_match = 1.0 if processed_norm == gt_norm else 0.0
    
    return {
        'exact_match': exact_match,
        'im': {
            'tp': im_result['tp'],
            'fn': im_result['fn'],
            'total': im_result['total'],
            'recall': im_result['recall'],
            'precision': im_precision,
            'f1': im_f1,
            'details': im_result['details']
        },
        'rm_r': {
            'tp': rm_r_result['tp'],
            'fn': rm_r_result['fn'],
            'total': rm_r_result['total'],
            'recall': rm_r_result['recall'],
            'precision': rm_r_precision,
            'f1': rm_r_f1,
            'details': rm_r_result['details']
        },
        'd': {
            'tp': d_token_metrics['tp'],
            'fn': d_token_metrics['fn'],
            'fp': d_token_metrics['fp'],
            'total': d_token_metrics['total'],
            'recall': d_token_metrics['recall'],
            'precision': d_token_metrics['precision'],
            'f1': d_token_metrics['f1'],
            'details': d_result['details'],
            'tag_tp': d_result['tp'],
            'tag_fn': d_result['fn'],
            'tag_total': d_result['total'],
            'residual_tokens': d_token_metrics['residual_tokens'],
            'fp_tokens': d_token_metrics['fp_tokens']
        },
        'false_positives': {
            'count': fp_result['fp_count'],
            'contents': fp_result['fp_contents']
        },
        'ground_truth': ground_truth,
        'processed_text': processed_text
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("annotated_path", help="Annotated transcript path")
    parser.add_argument("processed_path", help="Model-processed transcript path")
    args = parser.parse_args()

    results = evaluate_v2(args.annotated_path, args.processed_path)
    print_evaluation_report(results)
