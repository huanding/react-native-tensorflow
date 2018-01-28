#include <UIKit/UIKit.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"

@interface ImageRecognizer: NSObject
- (id) initWithData:(NSString *)model labels:(NSString *)labels;
- (NSArray *) recognizeImage:(NSString *)image maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold;
@end
