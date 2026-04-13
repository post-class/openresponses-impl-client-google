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

### ツールのフォローアップには同じクライアントインスタンスが必要です

Geminiのツールフォローアップは`function_response`を使用し、元の関数名が必要です。  
OpenResponsesの`function_call_output`は`call_id`のみを保持するため、このクライアントはメモリ内マッピングを保持します：

- `call_id -> 関数名`

影響：

- ツールのフォローアップは同じ`GeminiResponsesClient`インスタンスで行う必要があります。
- `previous_response_id`からのステートレスなリプレイは実装されていません。
- 未知の`call_id`の`function_call_output`が到着した場合、クライアントは`ValueError`を発生させます。

### メディア入力はベストエフォートで正規化されます

`input_image`、`input_file`、`input_video`について：

- `data:` URIはデコードされ、インラインバイトとして送信されます
- その他のURIはURIベースのパーツとして送信されます
- MIMEタイプはURIまたはファイル名から推測されます
- フォールバックのMIMEタイプは`application/octet-stream`です

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