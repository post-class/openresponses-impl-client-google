# openresponses-impl-client-google

OpenResponsesインターフェースを実装したGoogle Gemini用のPythonクライアントライブラリです。

## 概要

このパッケージは`BaseResponsesClient`互換のGeminiクライアントを提供します：

- 非ストリーミング: `ResponseResource`を返却
- ストリーミング: `AsyncIterator[ResponseStreamingEvent]`を返却
- リクエストモデル: `CreateResponseBody`

内部ではGoogle GenAI SDK（`google-genai`）を使用していますが、リクエストとレスポンスを`openresponses-impl-core`のOpenResponsesモデルに正規化します。

## インストール

```bash
uv add openresponses-impl-client-google
```

依存関係：

- Python `>=3.12`
- `google-genai>=1.72.0`
- `openresponses-impl-core>=0.1.0`

## 基本的な使い方

```python
from openresponses_impl_core.models.openresponses_models import CreateResponseBody
from openresponses_impl_client_google.client.gemini_responses_client import GeminiResponsesClient


client = GeminiResponsesClient(
    model="gemini-3-flash-preview",
    google_api_key="YOUR_API_KEY",  # GOOGLE_API_KEY / GEMINI_API_KEYが設定されている場合は省略可
)

payload = CreateResponseBody(
    input="こんにちは",
    stream=False,
)

response = await client.create_response(payload=payload)
print(response.output)
```

ストリーミング：

```python
payload = CreateResponseBody(
    input="再帰について簡潔に説明してください。",
    stream=True,
)

event_stream = await client.create_response(payload=payload)
async for event in event_stream:
    print(event.type)
```

## 特殊な処理

### システムと開発者の指示はマージされます

GeminiはOpenResponsesの`instructions`、`system`メッセージ、`developer`メッセージをOpenAI Responses APIと同じ形式では受け取りません。

このクライアントは以下をマージします：

- `payload.instructions`
- `role="system"`の`input`アイテム
- `role="developer"`の`input`アイテム

これらを単一のGemini `system_instruction`文字列にまとめます。

さらに、同じ `GeminiResponsesClient` インスタンス内では、`system` /
`developer` メッセージ由来の sticky なコンテキストだけをキャッシュします。
`instructions` 自体はキャッシュせず、毎リクエストの値だけを使います。
Gemini に送る直前に、毎ターン以下の順で `system_instruction` を再構成します。

- そのターンの `payload.instructions`
- キャッシュ済みの sticky な `system/developer` コンテキスト

そのため、follow-up ターンで `instructions` が省略された場合、前ターンの
`instructions` は引き継がれません。一方で `system/developer` が省略された
場合は、前回の sticky コンテキストを再利用します。後続ターンで新しい
`system/developer` が来た場合は、追記ではなく置換します。

### ツールのフォローアップには同じクライアントインスタンスが必要です

Geminiのツールフォローアップは`function_response`を使用し、元の関数名が必要です。  
OpenResponsesの`function_call_output`は`call_id`のみを保持するため、このクライアントはメモリ内マッピングを保持します：

- `call_id -> 関数名`

また、Geminiの関数呼び出しには`thought_signature`が付くことがあり、このクライアントはそれもGemini固有の状態として別途メモリ内マッピングで保持します：

- `call_id -> thought_signature`

影響：

- ツールのフォローアップは同じ`GeminiResponsesClient`インスタンスで行う必要があります。
- `previous_response_id`からのステートレスなリプレイは実装されていません。
- 未知の`call_id`の`function_call_output`が到着した場合、クライアントは`ValueError`を発生させます。

### `thought_signature` は正規化して再利用されます

Geminiは`function_call`パートに`thought_signature`を返すことがあります。このフィールドは汎用のOpenResponsesスキーマには存在しないため、このクライアントではGemini固有拡張として保持します：

```json
{
  "type": "function_call",
  "call_id": "call_1",
  "name": "lookup_weather",
  "arguments": "{\"city\":\"Tokyo\"}",
  "extensions": {
    "google": {
      "thought_signature": "c2lnbmF0dXJlLTEyMw"
    }
  }
}
```

特殊処理の内容：

- Geminiが`thought_signature`を`bytes`で返した場合でも、公開時にはパディングなしのURL-safe base64文字列へ正規化して`extensions.google.thought_signature`に格納します。
- クライアントは`call_id`単位で`thought_signature`をメモリ内にキャッシュします。
- 後続ターンで同じ`function_call`をOpenResponses入力として再送する際、`extensions.google.thought_signature`があればGemini SDK向けの`bytes`へ復元して渡します。
- 再送時にその拡張が省略されていても、同じ`call_id`に対するキャッシュ済み値があればそれを再利用します。

影響：

- 複数ターンのツール実行フローでは、同じ`GeminiResponsesClient`インスタンスを継続利用してください。
- プロセス外にツール状態を保存する場合は、OpenResponsesアイテム本体に加えて`extensions.google.thought_signature`も保持してください。
- 不正な`thought_signature`ペイロードを受け取った場合、`ValueError`を発生させます。

### Gemini ネイティブのターン履歴をツールフォローアップ用にキャッシュします

Gemini の並列ツール呼び出しは、provider-native のターン構造を失って OpenResponses アイテムへ平坦化し、それを 1 件ずつ再構成して再送すると不安定になります。

特に Gemini は、1 回の model turn に複数の `function_call` part を返すことがありますが、`thought_signature` が先頭 part にしか付かないことがあります。この状態で後続ターンが各 call を別々の Gemini turn として再構築すると、Gemini から以下で拒否されることがあります。

- `400 INVALID_ARGUMENT`
- `Function call is missing a thought_signature`

これを避けるため、このクライアントは `GeminiResponsesClient` インスタンス内に Gemini ネイティブの `contents` 履歴をメモリ保持するようになりました。

現在の動作：

- 初回リクエストは従来どおり OpenResponses input から Gemini `contents` へ変換します。
- Gemini 応答の native turn (`candidate.content`) も同じメモリ履歴へ追加します。
- 後続の follow-up リクエストは「今回ターンの delta 入力」として扱います。
- 連続する OpenResponses `function_call` は 1 つの Gemini `ModelContent(parts=[...])` に再グループ化します。
- 連続する OpenResponses `function_call_output` は 1 つの Gemini `UserContent(parts=[...])` に再グループ化します。
- 実際に Gemini へ送るリクエストは「キャッシュ済み native history + 今回の delta」から組み立て、平坦化済み OpenResponses 履歴のステートレス再送は行いません。
- sticky な `system/developer` コンテキストは `contents` とは別に保持され、Gemini へ送る `system_instruction` は毎ターン `current instructions + cached sticky context` で再構成されます。

影響：

- 複数ターンの Gemini ツールループでは、同じ `GeminiResponsesClient` インスタンスを必ず再利用してください。
- follow-up ごとに新しい client を作る実装だと、Gemini のツール再送は依然として失敗する可能性があります。
- このパッケージでは `previous_response_id` を使った Gemini のステートレス再開は引き続き未実装です。
- `parallel_tool_calls` は OpenResponses レベルのフラグのままであり、安全性はその値ではなく Gemini ネイティブのターン構造保持で担保しています。

### メディア入力はベストエフォートで正規化されます

OpenResponsesの`input_image`、`input_file`、`input_video`は、以下のように自動的にGemini APIの形式に変換されます：

#### 1. `input_image` / `input_video` の Data URI形式（`data:`で始まるURI）
```python
# 例: data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...
Message(
    role="user",
    content=[
        InputImage(image_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...")
    ]
)
```
- Base64エンコードされたデータをデコードして、バイト列として送信
- Gemini APIの`types.Part.from_bytes(data=..., mime_type=...)`を使用
- **用途**: 小さい画像・動画をリクエストに直接埋め込む場合

#### 2. `input_file` の `file_data`（raw base64 データ）
```python
Message(
    role="user",
    content=[
        InputFile(
            filename="sample.pdf",
            file_data="JVBERi0xLjQKJ..."
        )
    ]
)
```
- `file_data` は `data:` URI ではなく raw base64 データとして扱われます
- Base64データをデコードして inline bytes として送信します
- Gemini APIの`types.Part.from_bytes(data=..., mime_type=...)`を使用します
- MIMEタイプは `filename` から推測し、推測できない場合は `application/octet-stream` にフォールバックします

#### 3. `input_file` の `file_url`、またはURIベースの `input_video`
```python
# 例: gs://bucket/video.mp4, https://example.com/image.jpg
Message(
    role="user",
    content=[
        InputVideo(video_url="gs://my-bucket/video.mp4")
    ]
)
```
- URIをそのまま参照として送信
- Gemini APIの`types.Part.from_uri(file_uri=..., mime_type=...)`を使用
- **用途**: GCS上のファイル、YouTube動画、外部URLの参照

#### 4. MIMEタイプの自動推測
- URIやファイル名の拡張子から自動的にMIMEタイプを推測（例: `.mp4` → `video/mp4`）
- 推測できない場合は`application/octet-stream`にフォールバック
- 警告ログが出力されます

#### 5. Files APIでアップロードしたファイル
```python
# Google GenAI SDKで事前アップロード
video_file = genai_client.files.upload(file="path/to/video.mp4")
# そのままinputに渡せる
payload = CreateResponseBody(input=[video_file, ...])
```
- アップロード済みの`File`オブジェクトをそのまま`input`に含められる
- 内部的にGemini APIが適切に処理

#### 動画入力の推奨方法（Files API）

長尺・大容量の動画ファイルを扱う場合は、Files APIで事前にアップロードしてから本ライブラリを使用することを推奨します：

```python
from google import genai
from openresponses_impl_core.models.openresponses_models import CreateResponseBody, Message
from openresponses_impl_client_google.client.gemini_responses_client import GeminiResponsesClient
import time

# 1. Google GenAI SDKで動画をアップロード
genai_client = genai.Client()
video_file = genai_client.files.upload(file="path/to/video.mp4")

# 2. 処理完了まで待機（動画は処理時間が必要な場合がある）
while True:
    video_file = genai_client.files.get(name=video_file.name)
    if video_file.state != "PROCESSING":
        break
    time.sleep(2)

# 3. OpenResponsesクライアントで解析
responses_client = GeminiResponsesClient(
    model="gemini-3-flash-preview",
    google_api_key="YOUR_API_KEY",
)

payload = CreateResponseBody(
    input=[
        video_file,  # アップロード済みのFileオブジェクトを直接渡す
        Message(role="user", content="この動画を要約して、重要ポイントを箇条書きで教えてください。")
    ],
    stream=False,
)

response = await responses_client.create_response(payload=payload)
print(response.output)

# 4. 後片付け（任意）
genai_client.files.delete(name=video_file.name)
```

**注意事項：**
- Files APIでアップロードしたファイルオブジェクトは、そのまま`input`配列に含めることができます
- 動画の処理完了（`PROCESSING` → `ACTIVE`）を待ってから使用してください
- 小さい動画の場合は、`data:` URIやGCS URIを使用することもできます

### 推論は近似的にマッピングされます

OpenResponsesの推論設定はGeminiと1:1でマッピングされません。

現在の動作：

- `reasoning.effort="none"` -> `thinking_budget=0`
- `reasoning.effort="low" | "medium" | "high"` -> Gemini `thinking_level`
- `reasoning.effort="xhigh"` -> 警告付きで`HIGH`にマッピング
- `reasoning.summary` -> 警告付きで無視

### レスポンスフィールドは部分的に合成されます

Geminiは`ResponseResource`と同一のオブジェクトを返さないため、一部のフィールドは以下から再構築されます：

- リクエストペイロード
- Geminiレスポンスメタデータ
- クライアント生成のフォールバックIDとタイムスタンプ

例：

- `id`はGeminiが`response_id`を返さない場合、合成レスポンスIDにフォールバックします
- `tools`、`tool_choice`、`text`、`service_tier`などのフィールドは有効なリクエストからエコーされます
- Gemini固有のメタデータは`metadata["gemini_*"]`に格納されます

## OpenResponses互換性に関する注意事項

このクライアントは意図的にベストエフォートです。OpenResponsesのパブリックインターフェースを維持しますが、すべてのフィールドがGeminiでネイティブに表現できるわけではありません。

### 十分にサポートされている機能

- プレーンテキスト入力
- ユーザー/アシスタントメッセージ履歴
- 関数ツール
- 非ストリーミングレスポンス
- 基本的なストリーミングテキストレスポンス
- Gemini `response_json_schema`を介したJSON-schema形式の構造化出力

### 変換を伴うサポート

- `instructions`、`system`、`developer` -> マージされた`system_instruction`
- `function_call` -> Gemini `function_call`
- `function_call_output` -> Gemini `function_response`
- 推論テキスト/思考パーツ -> OpenResponses `reasoning`
- 使用量フィールド -> Gemini `usage_metadata`からマッピング

### 警告付きで無視されるフィールド

これらのフィールドは可能な場合、正規化された`ResponseResource`に保持されますが、機能的なリクエスト制御としてGeminiに送信されません：

- `previous_response_id`
- `store`
- `background`
- `parallel_tool_calls`
- `max_tool_calls`
- `truncation`
- `include`
- `safety_identifier`
- `prompt_cache_key`

### サポートされていない、または部分的にサポートされている動作

- `previous_response_id`
  - Geminiリクエスト実行では無視されます。
  - インターフェース互換性のためにのみ、正規化されたレスポンスに保持されます。

- 汎用ツール
  - `type="function"`ツールのみが変換されます。
  - 関数以外のツールは警告付きで無視されます。

- Gemini相当物のないアイテムタイプ
  - `item_reference`は警告付きで無視されます。
  - OpenResponses入力の`reasoning`アイテムは警告付きで無視されます。

- 正確なツール選択の忠実性
  - クライアントはツール選択をGemini関数呼び出し設定にベストエフォートで変換します。
  - OpenResponses/Coreモデルのシリアライゼーションは、Geminiクライアントが見る前に、一部の`tool_choice`形状に対してすでに損失がある可能性があります。

## ストリーミングセマンティクス

ストリーミングは最小限のOpenResponsesイベントセットに正規化されます。

現在発行されるイベントファミリー：

- `response.created`
- `response.output_item.added`
- `response.content_part.added`
- `response.output_text.delta`
- `response.output_text.done`
- `response.content_part.done`
- `response.output_item.done`
- `response.reasoning.delta`
- `response.reasoning.done`
- `response.function_call_arguments.done`
- 終了イベント：
  - `response.completed`
  - `response.incomplete`
  - `response.failed`
- `error`

注意事項：

- Geminiストリームチャンクはイベント変換前に累積的にマージされます。
- デルタ計算はプレフィックスベースのベストエフォートです。
- Geminiが予期しないチャンク形状を発行した場合、クライアントは`error`イベントを発行する可能性があります。

## ステータスマッピング

Geminiの終了状態は以下のようにOpenResponsesステータスにマッピングされます：

- プロンプトブロック/候補なし -> `failed`
- `MAX_TOKENS` -> `reason="max_tokens"`の`incomplete`
- `STOP` -> `completed`
- 関数呼び出しを含むレスポンス -> `completed`
- その他のGemini終了理由 -> Gemini理由文字列を持つ`incomplete`

## 認証

`google_api_key=`を直接渡すか、Google SDKの環境変数解決に依存できます。

一般的な環境変数：

- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`

## ログ動作

このクライアントは致命的でない非互換性に対して警告を使用します。  
以下の場合に警告ログが表示されます：

- サポートされていないOpenResponsesフィールドが提供された場合
- 汎用ツールが提供された場合
- サポートされていないアイテム/コンテンツタイプが提供された場合
- MIMEタイプ推論が`application/octet-stream`にフォールバックした場合
- `reasoning.summary`がリクエストされた場合
- `reasoning.effort="xhigh"`がダウングレードされた場合

## テスト

以下のコマンドでテストを実行します：

```bash
UV_CACHE_DIR="$PWD/.uv_cache" uv run pytest -q
```

## まとめ

OpenResponsesインターフェースの背後でGeminiを使用したい場合にこのパッケージを使用してください。ただし、以下の点に注意してください：

- インターフェース互換ですが、ワイヤー互換ではありません
- いくつかのOpenResponses制御はエミュレートまたは無視されます
- ツールのフォローアップはクライアントローカルのメモリ内状態に依存します
- ストリーミングは完全なGemini-to-Responsesプロジェクションではなく、実用的なサブセットに正規化されます
