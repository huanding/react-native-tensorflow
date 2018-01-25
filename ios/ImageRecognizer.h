#include <UIKit/UIKit.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"

@interface ImageRecognizer: NSObject
- (id) initWithData:(NSString *)model labels:(NSString *)labels imageMean:(NSNumber *)imageMean imageStd:(NSNumber *)imageStd;
- (NSArray *) recognizeImage:(NSString *)image inputSize:(NSNumber *)inputSize maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold;
@end
